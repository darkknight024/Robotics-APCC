#!/usr/bin/env python3
"""
Combinatorial Search Plotting Module

Provides visualization functions for combinatorial search ranking results.
Generates bar charts showing best and worst knife pose rankings.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def generate_ranking_plot(
    results: List,  # List[AggregatedKnifePoseResult]
    output_path: str,
    robot_name: str,
    top_n: int = 10
) -> None:
    """
    Generate bar chart showing top-N best and worst knife poses.
    
    Args:
        results: Sorted list of aggregated results (best first)
        output_path: Path to save PNG
        robot_name: Robot name for title
        top_n: Number of poses to show on each end
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if len(results) < 2:
            logger.warning(f"Not enough results to generate plot for {robot_name}")
            return
        
        # Get best and worst
        n_show = min(top_n, len(results) // 2)
        if n_show < 1:
            n_show = 1
        
        best = results[:n_show]
        worst = results[-n_show:]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Best poses
        ax1 = axes[0]
        names_best = [
            r.knife_pose_id[:25] + '...' if len(r.knife_pose_id) > 25 else r.knife_pose_id 
            for r in best
        ]
        scores_best = [r.normalized_score for r in best]
        colors_best = plt.cm.Greens(np.linspace(0.4, 0.8, len(best)))
        
        ax1.barh(range(len(best)), scores_best, color=colors_best)
        ax1.set_yticks(range(len(best)))
        ax1.set_yticklabels(names_best)
        ax1.set_xlabel('Normalized Score (lower=better)')
        ax1.set_title(f'Top {n_show} Best Knife Poses')
        ax1.invert_yaxis()
        ax1.set_xlim(0, 1)
        
        # Worst poses
        ax2 = axes[1]
        names_worst = [
            r.knife_pose_id[:25] + '...' if len(r.knife_pose_id) > 25 else r.knife_pose_id 
            for r in worst
        ]
        scores_worst = [r.normalized_score for r in worst]
        colors_worst = plt.cm.Reds(np.linspace(0.4, 0.8, len(worst)))
        
        ax2.barh(range(len(worst)), scores_worst, color=colors_worst)
        ax2.set_yticks(range(len(worst)))
        ax2.set_yticklabels(names_worst)
        ax2.set_xlabel('Normalized Score (lower=better)')
        ax2.set_title(f'Top {n_show} Worst Knife Poses')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)
        
        plt.suptitle(f'Knife Pose Ranking for {robot_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ranking plot: {output_path}")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
