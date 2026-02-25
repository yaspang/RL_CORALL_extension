"""
Script to add an RLlib callback that logs episode metrics for evaluation
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CORALL_Colav_TrainingCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, episode, worker=None, **kwargs):
        """Called at the end of an episode in RLlib 2.x"""
        
        if episode is None:
            return

        # episode._agent_to_last_info is a dict per agent (for multi-agent env)
        last_infos = getattr(episode, "_agent_to_last_info", {}) or {}

        n = 0
        n_success = 0
        n_collision = 0
        max_risks = []
        min_dcpas = []
        min_dists = []

        for agent_id, info in last_infos.items():
           if not isinstance(info, dict):
               continue
           
           n += 1
           n_success += int(bool(info.get("success", False)))
           n_collision += int(bool(info.get("collision", False)))

           if "max_risk" in info: max_risks.append(float(info["max_risk"]))
           if "min_dcpa" in info: min_dcpas.append(float(info["min_dcpa"]))
           if "min_dist" in info: min_dists.append(float(info["min_dist"]))


        # episode-level rates
        if n > 0: 
            episode.custom_metrics["success_rate"] = n_success / n
            episode.custom_metrics["collision_rate"] = n_collision / n
        
        # safety summaries
        if max_risks: episode.custom_metrics["max_risk_mean"] = sum(max_risks) / len(max_risks)
        if min_dcpas: episode.custom_metrics["min_dcpa_mean"] = sum(min_dcpas) / len(min_dcpas)
        if min_dists: episode.custom_metrics["min_dist_mean"] = sum(min_dists) / len(min_dists)