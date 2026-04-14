"""
Script to add an RLlib callback that logs episode metrics for evaluation
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

class CORALL_Colav_TrainingCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, episode, worker=None, **kwargs):
        """Called at the end of an episode in RLlib 2.x"""
        
        if episode is None:
            return
        
        # RLlib version compatability check: episode should have _agent_to_last_info dict with per-agent info from env.step() if using 
        # RLlib 2.x and env is returning info dicts
        ## episode._agent_to_last_info is a dict per agent (for multi-agent env)
        last_infos = getattr(episode, "_agent_to_last_info", {}) or {}

        if not last_infos:
            last_infos = getattr(episode, "last_infos", {}) or {}
        
        if not last_infos:
            return

        n = 0
        n_success = 0
        n_collision = 0

        path_lengths = []
        min_dcpas = []
        risk_exposures = []
        completion_times = []

        max_risks = []
        min_dists = []

        for agent_id, info in last_infos.items():
           if not isinstance(info, dict):
               continue

           if "max_risk" in info: max_risks.append(float(info["max_risk"]))
           if "min_dist" in info: min_dists.append(float(info["min_dist"]))

           epm = info.get("episode_metrics", None)

           if not isinstance(epm, dict):
                continue
            
           n += 1
           
           path_lengths.append(float(epm.get("path_length_m", np.nan))) 
           min_dcpas.append(float(epm.get("min_dcpa_m", np.nan)))
           risk_exposures.append(float(epm.get("risk_exposure", np.nan)))

           success_i = int(bool(epm.get("success", 0)))
           n_success += success_i
           collision_i = int(bool(epm.get("collision", 0)))
           n_collision += collision_i

           ct = epm.get("completion_time_s", None)
           if ct is not None: completion_times.append(float(ct))


        # episode-level rates
        if n > 0: 
            episode.custom_metrics["success_rate"] = n_success / n
            episode.custom_metrics["collision_rate"] = n_collision / n

            # collision rate per episode: whether any agent collided during the episode
            episode.custom_metrics["collision_rate_episode"] = float(n_collision > 0)
        
        # safety summaries
        if max_risks: episode.custom_metrics["max_risk_mean"] = float(np.mean(max_risks))
        if min_dists: episode.custom_metrics["min_dist_mean"] = float(np.mean(min_dists))

        # performance summaries
        if path_lengths: episode.custom_metrics["path_length_m_mean"] = float(np.nanmean(path_lengths))
        if min_dcpas: episode.custom_metrics["min_dcpa_m_mean"] = float(np.nanmean(min_dcpas))
        if risk_exposures: episode.custom_metrics["risk_exposure_mean"] = float(np.nanmean(risk_exposures))
        if completion_times: episode.custom_metrics["completion_time_s_mean"] = float(np.nanmean(completion_times))

