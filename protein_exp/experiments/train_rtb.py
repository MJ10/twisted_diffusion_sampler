# Import feynman_kac_pf from a level down
import sys
import os
import logging
import subprocess as sp
import torch
import hydra
import numpy as np
import pandas as pd
import time
import tree
import shutil
from data import utils as du
from omegaconf import DictConfig, OmegaConf

import wandb
from smc_utils_prot import feynman_kac_pf, smc_utils
from motif_scaffolding import save_motif_segments, twisting
from motif_scaffolding import utils as mu
from data import all_atom
from openfold.utils import rigid_utils as ru

from experiments import inference_motif_scaffolding
from experiments import utils as eu
from openfold.utils.rigid_utils import Rigid
from openfold.utils import rigid_utils as ru
from analysis import utils as au
from peft import LoraConfig, get_peft_model


def align_outputs_on_motif(sampler, sampler_cache, idx, rigids_motif, insert_motif_at_t0=False):
    """align_outputs_on_motif aligns the outputs of the sampler on the motif.

    if insert_motif is True, then the motif is inserted into the protein.

    """
    # align sample output on the motif
    sample_output = sampler_cache["model_out"]
    all_motif_locations = sampler_cache["all_motif_locations"]
    all_rots = sampler_cache["all_rots"]
    psis = torch.zeros_like(rigids_motif[:, :2])
    atom37_motif = all_atom.compute_backbone(
        ru.Rigid.from_tensor_7(rigids_motif), psis)[0]

    max_log_p_idx = sample_output['max_log_p_idx'][0]
    motif_locations_b = all_motif_locations[max_log_p_idx]
    sampler_cache['prot_traj'][0, idx] = au.align_on_motif_bb(
        torch.tensor(sampler_cache['prot_traj'][0, idx]),
        motif_locations_b,
        atom37_motif,
        insert_motif=insert_motif_at_t0)
    if not "aux_traj" in sampler_cache:
        return sampler_cache
    all_bb_prots = sampler.PF_cache['aux_traj']['all_bb_prots']
    all_bb_0_pred = sampler.PF_cache['aux_traj']['all_bb_0_pred']
    max_log_p_idx_by_t = sampler.PF_cache['max_log_p_idx_by_t']
    for p in range(all_bb_prots[0].shape[0]):
        max_log_p_idx = sample_output['max_log_p_idx'][p]
        for t in range(len(all_bb_prots)):
            # Instead use most likely rot and offset to align.

            # Center on motif and rotate to be on the motif
            f_idx = max_log_p_idx_by_t[t][p]
            motif_locations_tp = all_motif_locations[f_idx]

            # Since when all_motif_locations is computed, it is of length num_rots*num_offsets,
            # with blocks of num_rots for each offset, we get the index of the rotation like this.
            rot_idx = f_idx % all_rots.shape[0]
            R = all_rots[rot_idx].cpu().numpy()

            # Center so that most likely prediction on motif is at origin
            pred_bb = torch.tensor(all_bb_0_pred[t][p])
            motif_pred = torch.cat([pred_bb[st:(end+1)] for st, end in motif_locations_tp], dim=0)
            motif_pred_CA = motif_pred[1]
            motif_com = torch.mean(motif_pred_CA, dim=0).numpy()
            all_bb_0_pred[t][p] = all_bb_0_pred[t][p] - motif_com
            all_bb_prots[t][p] = all_bb_prots[t][p] - motif_com

            # Rotate so that most likely prediction on motif is aligned with motif
            all_bb_0_pred[t][p] = np.einsum('ij,bAj->bAi', R, all_bb_0_pred[t][p])
            all_bb_prots[t][p] = np.einsum('ij,bAj->bAi', R, all_bb_prots[t][p])

    sampler.PF_cache['aux_traj']['all_bb_prots'] = all_bb_prots
    sampler.PF_cache['aux_traj']['all_bb_0_pred'] = all_bb_0_pred

def save_aux_traj(sampler, sample_dir, aatype=None, diffuse_mask=None):
    """save_aux_traj saves the auxiliary trajectory of the particle filter,
     aligned to the motif (if provided).

     Args:
        diffuse_mask with shape [T, P, L]
    """
    if diffuse_mask is None:
        diffuse_mask = np.ones(all_bb_prots.shape[2])
    # Flip trajectory so that it starts from t=0.
    # This helps visualization.
    flip = lambda x: np.flip(np.stack(x), (0,))
    all_bb_prots = flip(sampler.PF_cache['aux_traj']['all_bb_prots'])
    all_bb_0_pred = flip(sampler.PF_cache['aux_traj']['all_bb_0_pred'])
    diffuse_mask = flip(diffuse_mask)

    for p in range(all_bb_0_pred.shape[1]):
        au.save_traj(
            all_bb_prots[:, p],
            all_bb_0_pred[:, p],
            aatype=aatype,
            diffuse_mask=diffuse_mask[:, p],
            output_dir=sample_dir,
            prefix="P{p:03d}_aux_".format(p=p)
        )


def init_particle_filter(sampler, motif_contig_info, P=4):
    sampler.PF_cache = {}
    motif_segments = [torch.tensor(motif_segment, dtype=torch.float64) for motif_segment in motif_contig_info['motif_segments']]
    rigids_motif = eu.remove_com_from_tensor_7(
        torch.cat([motif_segment.to(sampler.device) for motif_segment in motif_segments], dim=0))
    sampler.PF_cache["rigids_motif"] = rigids_motif

    if sampler._infer_conf.motif_scaffolding.use_contig_for_placement:
        num_DOF = sampler._infer_conf.motif_scaffolding.num_rots*sampler._infer_conf.motif_scaffolding.max_offsets
        assert num_DOF == 1, f"sampling using contig supported only for no DOF {num_DOF}"

    assert not sampler._infer_conf.motif_scaffolding.use_replacement, "replacement not supported for PF"
    # Check if number of possible rotations and translations is 1 and sample motif if so
    if sampler._infer_conf.motif_scaffolding.num_rots == 1 and sampler._infer_conf.motif_scaffolding.max_offsets == 1:
        motif_locations, length = sampler.motif_locations_and_length(
            motif_segments, motif_contig_info, P)
    else:
        motif_locations, length = None, motif_contig_info['length_fixed']
    sampler.PF_cache["motif_locations"] = motif_locations
    sampler.PF_cache["length"] = length

    # Compute vectorized function for degrees of freedom computation
    F, all_motif_locations, all_rots = twisting.motif_offsets_and_rots_vec_F(
        length, motif_segments, motif_locations=motif_locations, num_rots=sampler._infer_conf.motif_scaffolding.num_rots,
        device=sampler.device,
        max_offsets=sampler._infer_conf.motif_scaffolding.max_offsets,
        return_rots=True)
    sampler.PF_cache["F"] = F
    sampler.PF_cache["all_motif_locations"] = all_motif_locations
    sampler.PF_cache["all_rots"] = all_rots


    # Initialize sample_feats
    res_mask = np.ones([P, length])
    fixed_mask = np.zeros_like(res_mask)
    sample_feats = {
        'res_mask': res_mask,
        'seq_idx': torch.arange(1, length+1).repeat(P, 1),
        'fixed_mask': fixed_mask,
        'torsion_angles_sin_cos': np.zeros((P, length, 7, 2)),
        'sc_ca_t': np.zeros((P, length, 3)),
        'motif_locations': motif_locations, # list of length batch_size of lists of motif segment locations
        't_placeholder': torch.ones((P,)),
        'rigids_motif': rigids_motif,
    }

    # Add move to torch and GPU
    sample_feats = tree.map_structure(lambda x: x if (x is None or torch.is_tensor(x)) else torch.tensor(x), sample_feats)
    sample_feats = tree.map_structure(lambda x: x if x is None else x.to(sampler.device), sample_feats)
    sampler.PF_cache["sample_feats"] = sample_feats
    sampler.PF_cache["motif_contig_info"] = motif_contig_info


def log_and_validate_PF_output(
    sampler, sampler_cache, sample_ids,
    keep_motif_seq=False, insert_motif_at_t0=False
    ):
    # Run self-consistency validation on the first particle
    sample_output = sampler_cache["model_out"]
    all_motif_locations = sampler_cache["all_motif_locations"]
    length = sampler_cache["length"]
    rigids_motif = sampler_cache["rigids_motif"]
    motif_contig_info = sampler_cache["motif_contig_info"]

    # Compute / extract motif mask from output
    # TOD0: Adapt this to work with SMCDiff?
    for i in range(len(sample_ids)):
        # Check if we've already run this sample
        sample_id = sample_ids[i]
        max_log_p_idx = sample_output['max_log_p_idx'][i]
        motif_locations_b = all_motif_locations[max_log_p_idx]

        # Set positions of the motif in the sequence to the prescribed amino acids
        aatype = np.zeros([length], dtype=np.int32) # Default to Alanine
        for j, (st, end) in enumerate(motif_locations_b):
            aatype[st:end+1] = motif_contig_info['aatype'][j]
        sample_dir = os.path.join(sampler._output_dir, f'length_{sampler_cache["length"]}', f'sample_{sample_id}')
        if os.path.isdir(sample_dir):
            print("Skipping sample, prev sample already run", sample_id)
            continue
        os.makedirs(sample_dir, exist_ok=True)
        # Save motif locations to file
        motif_segments_string = ",".join([f"{st}_{end}" for st, end in motif_locations_b])
        motif_segments_fn = os.path.join(sample_dir, f"motif_segments_{sample_id}.txt")
        with open(motif_segments_fn, "w") as f:
            f.write(motif_segments_string)

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((np.ones(length) * 100)[:, None], (1, 37))
        
        
        sample_path = os.path.join(sample_dir, 'sample')

        sampler_cache = align_outputs_on_motif(sampler, sampler_cache, i, rigids_motif,
                insert_motif_at_t0=insert_motif_at_t0)

        pdb_path = au.write_prot_to_pdb(
            sampler_cache['prot_traj'][:, i][0],
            sample_path,
            aatype=aatype,
            b_factors=b_factors
        )
        
        # Run ProteinMPNN
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        os.makedirs(sc_output_dir, exist_ok=True)
        shutil.copy(pdb_path, os.path.join(
            sc_output_dir, os.path.basename(pdb_path)))
        
    if sampler._infer_conf.aux_traj:
        T = len(sampler.PF_cache['aux_traj']['all_bb_prots'])
        P = sampler.PF_cache['prot_traj'][0].shape[0]
        diffuse_mask = np.zeros([T, P, length])
        # set motif positions to 1
        max_log_p_idx_by_t = sampler.PF_cache['max_log_p_idx_by_t']
        for p in range(P):
            max_log_p_idx = sample_output['max_log_p_idx'][p]
            for t in range(T):
                motif_locations_tp = all_motif_locations[max_log_p_idx_by_t[t][p]]
                for st, end in motif_locations_tp:
                    diffuse_mask[t, p, st:end+1] = 1
        save_aux_traj(sampler, sample_dir, aatype, diffuse_mask=diffuse_mask)



def run_particle_filter(sampler, motif_contig_info, P=4, sample_id=0,
    keep_motif_seq=False, verbose=False, insert_motif_at_t0=False
 ):

    # First run init -- this does a few things:
    # * chooses the length of the backbones (# of residues)
    # * sets the degrees of freedom (if applicable) / motif locations
    init_particle_filter(sampler, motif_contig_info, P=P)

    # Check if we've already run this sample
    sample_dir = os.path.join(sampler._output_dir, f'length_{sampler.PF_cache["length"]}', f'sample_{sample_id}')
    if os.path.isdir(sample_dir):
        print("Skipping sample, prev sample already run", sample_id)
        return
    os.makedirs(sample_dir, exist_ok=True)


    # Set transition kernels and potential fucntion
    M = sampler.M
    G = sampler.G
    T = sampler._diff_conf.num_t
    resample = smc_utils.resampling_function(
        ess_threshold=sampler._infer_conf.particle_filtering.resample_threshold,
        verbose=verbose)

    # Run particle filter
    sampler._log.info(f'Running particle filter')
    xts, log_w, resample_indices_trace, ess_trace, log_w_trace  = feynman_kac_pf.smc_FK(
        M, G, resample, T, P, verbose=verbose)

    # Run self-consistency validation on the first particle
    log_and_validate_PF_output(sampler, log_w_trace, ess_trace, sample_id, sample_dir=sample_dir,
        keep_motif_seq=keep_motif_seq, insert_motif_at_t0=insert_motif_at_t0)

def likelihood_fn(rigids_pred, rigids_obs, sampler, F, sigma):
    """
    Based on https://github.com/blt2114/twisted_diffusion_sampler/blob/main/protein_exp/motif_scaffolding/twisting.py#L234
    """
    diffuser = sampler.exp.diffuser

    R_pred = Rigid.from_tensor_7(rigids_pred).get_rots().get_rot_mats().to(torch.device(sampler.device), rigids_pred.dtype)
    x_pred = Rigid.from_tensor_7(rigids_pred).get_trans().to(torch.device(sampler.device))

    R_pred, x_pred = F(R_pred, x_pred)
    rigids_obs = rigids_obs.unsqueeze(0)
    R_obs = Rigid.from_tensor_7(rigids_obs).get_rots().get_rot_mats().to(torch.device(sampler.device), rigids_pred.dtype)
    x_obs = Rigid.from_tensor_7(rigids_obs).get_trans().to(torch.device(sampler.device))

    # Compute term likelihood term for rotations
    log_p = 0.
    # Frobenius norm approximation to tangent normal density
    log_p += -((R_pred - R_obs[None, None]).pow(2)/(sigma)).sum(dim=[-3, -2, -1])

    # Compute term likelihood term for translations
    # scale down x_pred and x_obs
    x_pred = diffuser._r3_diffuser._scale(x_pred)
    x_obs = diffuser._r3_diffuser._scale(x_obs)

    log_p += -(((x_obs - x_pred)**2)/(sigma)).sum(dim=[-1, -2])

    return log_p.logsumexp(dim=-1), log_p.argmax(dim=-1)


def sample_fn(posterior_sampler, prior_sampler, motif_contig_info, batch_size, detach_freq=0.7, 
           keep_motif_seq=False, verbose=False, insert_motif_at_t0=False):
    # posterior_sampler.PF_cache = {}
    motif_segments = [torch.tensor(motif_segment, dtype=torch.float64) for motif_segment in motif_contig_info['motif_segments']]
    rigids_motif = eu.remove_com_from_tensor_7(
        torch.cat([motif_segment.to(posterior_sampler.device) for motif_segment in motif_segments], dim=0))
    # posterior_sampler.PF_cache["rigids_motif"] = rigids_motif

    if posterior_sampler._infer_conf.motif_scaffolding.use_contig_for_placement:
        num_DOF = posterior_sampler._infer_conf.motif_scaffolding.num_rots*posterior_sampler._infer_conf.motif_scaffolding.max_offsets
        assert num_DOF == 1, f"sampling using contig supported only for no DOF {num_DOF}"

    assert not posterior_sampler._infer_conf.motif_scaffolding.use_replacement, "replacement not supported for PF"
    
    # Check if number of possible rotations and translations is 1 and sample motif if so
    if posterior_sampler._infer_conf.motif_scaffolding.num_rots == 1 and posterior_sampler._infer_conf.motif_scaffolding.max_offsets == 1:
        motif_locations, length = posterior_sampler.motif_locations_and_length(
            motif_segments, motif_contig_info, batch_size)
    else:
        motif_locations, length = None, motif_contig_info['length_fixed']
    # posterior_sampler.PF_cache["motif_locations"] = motif_locations
    # posterior_sampler.PF_cache["length"] = length

    # Compute vectorized function for degrees of freedom computation
    F, all_motif_locations, all_rots = twisting.motif_offsets_and_rots_vec_F(
        length, motif_segments, motif_locations=motif_locations, num_rots=posterior_sampler._infer_conf.motif_scaffolding.num_rots,
        device=posterior_sampler.device,
        max_offsets=posterior_sampler._infer_conf.motif_scaffolding.max_offsets,
        return_rots=True)


    # Initialize sample_feats
    res_mask = np.ones([batch_size, length])
    fixed_mask = np.zeros_like(res_mask)
    sample_feats = {
        'res_mask': res_mask,
        'seq_idx': torch.arange(1, length+1).repeat(batch_size, 1),
        'fixed_mask': fixed_mask,
        'torsion_angles_sin_cos': np.zeros((batch_size, length, 7, 2)),
        'sc_ca_t': np.zeros((batch_size, length, 3)),
        'motif_locations': motif_locations, # list of length batch_size of lists of motif segment locations
        't_placeholder': torch.ones((batch_size,)),
        'rigids_motif': rigids_motif,
    }

    # Add move to torch and GPU
    sample_feats = tree.map_structure(lambda x: x if (x is None or torch.is_tensor(x)) else torch.tensor(x), sample_feats)
    sample_feats = tree.map_structure(lambda x: x if x is None else x.to(posterior_sampler.device), sample_feats)

    ref_sample = posterior_sampler.diffuser.sample_ref(
        n_samples=length * batch_size,
        as_tensor_7=True)
    rigid_t = Rigid.from_tensor_7(ref_sample['rigids_t'].reshape(batch_size, length, 7))
    sample_feats['R_t'] = rigid_t.get_rots().get_rot_mats().to(torch.float64)
    sample_feats['trans_t'] = rigid_t.get_trans().to(torch.float64)
    sample_feats['rigids_t'] = rigid_t.to_tensor_7().to(torch.float64)
    # Move all tensors in sample_feats onto device
    for k, v in sample_feats.items():
        if isinstance(v, torch.Tensor):
            sample_feats[k] = v.to(posterior_sampler.device)

    t_cts = 1.0
    self_conditioning = posterior_sampler.exp._model_conf.embed.embed_self_conditioning and not posterior_sampler._infer_conf.motif_scaffolding.no_self_conditioning
    if self_conditioning:
        sample_feats = posterior_sampler.exp._set_t_feats(
            sample_feats, t_cts, sample_feats['t_placeholder'])
        sample_feats = posterior_sampler.exp._self_conditioning(sample_feats)
        # posterior_sampler.PF_cache['sample_feats'] = sample_feats
    
    xt = rigid_t.to_tensor_7().cpu().detach()

    num_t = posterior_sampler._diff_conf.num_t
    T = num_t
    min_t = posterior_sampler._diff_conf.min_t
    dt = 1/num_t

    log_pf_posterior = 0.0
    log_pf_prior = 0.0

    ts = range(T, -1, -1)
    for t in ts:
        # print(t)
        t_cts = np.linspace(min_t, 1.0, num_t+1)[t]

        # Extract and update sample feats
        for k, v in sample_feats.items():
            if isinstance(v, torch.Tensor): sample_feats[k] = v.detach()

        # Update sample feats with new rigids and t feature
        sample_feats['rigids_t'] = xt.to(posterior_sampler.device)
        xt_rigids = Rigid.from_tensor_7(xt)
        sample_feats['R_t'] = xt_rigids.get_rots().get_rot_mats().to(posterior_sampler.device).to(torch.float64)
        sample_feats['trans_t'] = xt_rigids.get_trans().to(posterior_sampler.device).to(torch.float64)
        sample_feats = posterior_sampler.exp._set_t_feats(sample_feats, t_cts, sample_feats['t_placeholder'])

        # Run model
        model_out = posterior_sampler.exp.model(
            sample_feats, F=F,
            use_twisting=False,
        )
        with torch.no_grad():
            prior_out = prior_sampler.exp.model(
                sample_feats, F=F,
                use_twisting=False,
            )
        self_conditioning = posterior_sampler.exp._model_conf.embed.embed_self_conditioning and not posterior_sampler._infer_conf.motif_scaffolding.no_self_conditioning
        if self_conditioning: sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]

        rigid_pred = model_out['rigids']

        diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
        rigids_t, _ = posterior_sampler.exp.diffuser.reverse(
            rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
            rot_score=model_out['rot_score'],
            trans_score=model_out['trans_score'],
            diffuse_mask=diffuse_mask,
            t=t_cts,
            dt=dt,
            noise_scale=posterior_sampler._diff_conf.noise_scale,
            return_log_p_sample=True,
        )

        if posterior_sampler._infer_conf.aux_traj:
            # Calculate x0 prediction derived from score predictions.
            if not "aux_traj" in posterior_sampler.PF_cache:
                posterior_sampler.PF_cache['aux_traj'] = {'all_bb_0_pred': [], 'all_bb_prots': []}
                posterior_sampler.PF_cache['max_log_p_idx_by_t'] = []
            posterior_sampler.PF_cache['aux_traj']['all_bb_0_pred'].append(du.move_to_np(all_atom.compute_backbone(
                ru.Rigid.from_tensor_7(rigid_pred), model_out['psi_pred'])[0]))
            posterior_sampler.PF_cache['aux_traj']['all_bb_prots'].append(du.move_to_np(all_atom.compute_backbone(
                rigids_t, model_out['psi_pred'])[0]))
            posterior_sampler.PF_cache['max_log_p_idx_by_t'].append(model_out['max_log_p_idx'])


        # If the last step, return the model output not noised rigids
        if t==0:
            rigids_t = rigid_pred.cpu().detach()
        else:
            rigids_t = rigids_t.to_tensor_7().cpu().detach()

        # Add prot_traj and psi_pred with appropriate dimensios 0for validation
        prot_traj = du.move_to_np(
            all_atom.compute_backbone(ru.Rigid.from_tensor_7(
                rigids_t), model_out['psi_pred'])[0])[None]
        model_out['psi_pred'] = model_out['psi_pred'][None]
        
        xtp1 = rigids_t.detach()


        posterior_log_prob = posterior_sampler.exp.diffuser.reverse_log_prob(
            rigid_t=xtp1,
            rigid_tm1=xt,
            rot_score=model_out['rot_score'],
            trans_score=model_out['trans_score'],
            t=t_cts,
            dt=dt,
        ).to(posterior_sampler.device)

        with torch.no_grad():
            prior_log_prob = prior_sampler.exp.diffuser.reverse_log_prob(
                rigid_t=xtp1,
                rigid_tm1=xt,
                rot_score=prior_out['rot_score'],
                trans_score=prior_out['trans_score'],
                t=t_cts,
                dt=dt,
            ).to(posterior_sampler.device)
        log_pf_posterior += posterior_log_prob if np.random.uniform() > detach_freq else posterior_log_prob.detach()
        log_pf_prior += prior_log_prob
        xt = xtp1

    return xt, log_pf_posterior, log_pf_prior, {
        "prot_traj": prot_traj,
        "F": F,
        "all_motif_locations": all_motif_locations,
        "rigids_motif": rigids_motif,
        "all_rots": all_rots,
        "model_out": model_out,
        "motif_contig_info": motif_contig_info,
        "length": length,
    }


def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values


@hydra.main(version_base=None, config_path="../config", config_name="rtb")
def run(conf: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    # Read model checkpoint.
    logger.info('Starting inference')

    wandb.init(project="protein-inverse-prob", config=OmegaConf.to_container(conf, resolve=True), mode=conf.inference.wandb_mode)

    posterior_sampler = inference_motif_scaffolding.Sampler(conf)
    peft_model = get_peft_model(posterior_sampler.exp.model, LoraConfig(
        target_modules=["linear"],
        r=16,
        lora_alpha=32
    ))
    posterior_sampler.exp.replace_model(peft_model)
    posterior_sampler.exp.model.print_trainable_parameters()
    logger.info("GPU memory: " + str(get_gpu_memory()))
    
    prior_sampler = inference_motif_scaffolding.Sampler(conf)
    prior_sampler.exp.model.eval()
    torch.set_default_tensor_type('torch.FloatTensor')
    output_dir_stem = posterior_sampler._output_dir
    logger.info("GPU memory: " + str(get_gpu_memory()))
    trainable_params = sum(p.numel() for p in posterior_sampler.exp.model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {str(trainable_params)}")

    assert not posterior_sampler._infer_conf.motif_scaffolding.use_replacement, "use_replacement not implemented for particle filter"


    # Load motif test case details
    inpaint_df = pd.read_csv(posterior_sampler._infer_conf.motif_scaffolding.inpaint_cases_csv)
    contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(inpaint_df)
    if posterior_sampler._infer_conf.motif_scaffolding.test_name is not None:
        test_names = [posterior_sampler._infer_conf.motif_scaffolding.test_name]
        logger.info("running on test case: " + str(test_names))
    else:
        test_names = ["2KL8", "1BCF", "1PRW", "6EXZ_long"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        test_names = ["1PRW", "1QJG", "5TRV_short"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        #test_names = [name for name in contigs_by_test_case.keys()]

    # for test_name in test_names:
    test_name = posterior_sampler._infer_conf.motif_scaffolding.test_name
    logger.info("starting test case: " + test_name)
    motif_contig_info = contigs_by_test_case[test_name]

    posterior_sampler._output_dir = inference_motif_scaffolding.construct_output_dir(posterior_sampler, test_name, output_dir_stem)
    logger.info("output_dir: " + str(posterior_sampler._output_dir))
    os.makedirs(posterior_sampler._output_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(posterior_sampler.exp.model.parameters(), lr=conf.inference.learning_rate)
    optimizer.zero_grad()
    logger.info("Starting RTB training")

    for step in range(conf.inference.training_steps):
        if step < conf.inference.sigma_anneal_steps:
            if conf.inference.sigma_anneal_scheme == "linear":
                start_sigma = conf.inference.start_sigma
                end_sigma = conf.inference.end_sigma
                steps = conf.inference.sigma_anneal_steps
                curr_sigma = start_sigma + (end_sigma - start_sigma) * (step / steps)
            elif conf.inference.sigma_anneal_scheme == "exp":
                start_sigma = conf.inference.start_sigma
                end_sigma = conf.inference.end_sigma
                steps = conf.inference.sigma_anneal_steps
                curr_sigma = start_sigma * (end_sigma / start_sigma) ** (step / steps)
            elif conf.inference.sigma_anneal_scheme == "none":
                curr_sigma = conf.inference.start_sigma

        # Load in contig separately for each batch in order to sample a
        # different length for the motif in the case that the contig is used
        row = list(inpaint_df[inpaint_df.target==test_name].iterrows())[0][1]
        motif_contig_info = save_motif_segments.load_contig_test_case(row)
        x, log_pf_posterior, log_pf_prior, extras = sample_fn(posterior_sampler, 
                                                                prior_sampler, motif_contig_info, batch_size=conf.inference.batch_size,)
        
        logreward, _ = likelihood_fn(x, extras['rigids_motif'], posterior_sampler, extras["F"], curr_sigma)
        log_z_hat = (- log_pf_posterior + log_pf_prior + logreward).mean().detach()
        rtb_loss = ((log_pf_posterior - log_pf_prior - logreward + log_z_hat) ** 2).mean()
        wandb.log({
            "rtb_loss": rtb_loss.item(), 
            "logreward": logreward.mean().item()
        })
        rtb_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("step: ", step, "rtb_loss: ", rtb_loss.item(), " logreward: ", logreward.mean().item())

    logger.info("RTB training complete. Saving checkpoint")
    torch.save(posterior_sampler.exp.model.state_dict(), "model.pth")
    logger.info("Generating test samples")
    for i in range((conf.inference.motif_scaffolding.number_of_samples // conf.inference.eval_batch_size) + 1):
        ids = list(range(i*conf.inference.eval_batch_size, (i+1)*conf.inference.eval_batch_size))
        ids = [id for id in ids if id < conf.inference.motif_scaffolding.number_of_samples]
        with torch.no_grad():
            x, log_pf_posterior, log_pf_prior, extras = sample_fn(posterior_sampler, 
                                                                    prior_sampler, motif_contig_info, batch_size=len(ids),)
            logreward, idxs = likelihood_fn(x, extras['rigids_motif'], posterior_sampler, extras["F"], conf.inference.start_sigma)
            extras['model_out']['max_log_p_idx'] = idxs.squeeze().cpu()
            log_and_validate_PF_output(posterior_sampler, extras, ids)

    motif_segments = [
        torch.tensor(motif_segment, dtype=torch.float64)
        for motif_segment in motif_contig_info['motif_segments']
        ]
    rigids_motif = eu.remove_com_from_tensor_7(torch.cat([
        motif_segment.to(posterior_sampler.device) for motif_segment in motif_segments
        ], dim=0))
    
    logger.info("Loading folding model")
    posterior_sampler.load_folding_model()
    logger.info("Running self consistency eval")
    posterior_sampler.run_self_consistency(
        posterior_sampler._output_dir,
        rigids_motif=rigids_motif,
        use_motif_seq=posterior_sampler._infer_conf.motif_scaffolding.keep_motif_seq,
        motif_contig_info=motif_contig_info,
        )

if __name__ == '__main__':
    run()
