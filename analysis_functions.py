import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import itertools

def analyze_ipo_combinations(representation: Dict) -> Dict:
    """
    Analyze how different tutorials use different combinations of IPO elements for each subgoal.
    
    Returns:
    {
        phase: {
            step: {
                'combinations': List[Set[str]],  # List of unique IPO element combinations
                'frequency': Dict[frozenset, int],  # Frequency of each combination
                'diversity_score': float,  # Normalized measure of combination diversity
                'most_common': Set[str],  # Most frequently used combination
                'rare': List[Set[str]]  # Combinations used only once
            }
        }
    }
    """
    analysis = {}
    
    for phase in representation:
        analysis[phase] = {}
        for step in representation[phase]:
            combinations = []
            freq = defaultdict(int)
            
            # Collect all IPO combinations used in tutorials for this step
            for tutorial_url, tutorial_data in representation[phase][step].items():
                used_elements = set()
                for ipo_key in ['inputs', 'outputs', 'methods']:
                    if tutorial_data.get(ipo_key, []):
                        used_elements.add(ipo_key)
                combinations.append(used_elements)
                freq[frozenset(used_elements)] += 1
            
            # Calculate metrics
            unique_combinations = list(set(map(frozenset, combinations)))
            diversity_score = len(unique_combinations) / (2**3 - 1)  # Normalized by max possible combinations
            
            most_common = max(freq.items(), key=lambda x: x[1])[0]
            rare = [set(comb) for comb in freq if freq[comb] == 1]
            
            analysis[phase][step] = {
                'combinations': [set(c) for c in unique_combinations],
                'frequency': dict(freq),
                'diversity_score': diversity_score,
                'most_common': set(most_common),
                'rare': rare
            }
    
    return analysis

def analyze_information_fragmentation(representation: Dict) -> Dict:
    """
    Analyze how information is distributed across tutorials for each IPO element.
    
    Returns:
    {
        phase: {
            step: {
                ipo_key: {
                    'unique_info': Dict[str, List[str]],  # Tutorial URL -> unique information pieces
                    'complementary_pairs': List[Tuple[str, str]],  # Pairs of tutorials with complementary info
                    'conflicting_pairs': List[Tuple[str, str]],  # Pairs of tutorials with conflicting info
                    'completeness_scores': Dict[str, float],  # Tutorial URL -> completeness score
                    'distribution_stats': Dict  # Statistics about information distribution
                }
            }
        }
    }
    """
    analysis = {}
    
    for phase in representation:
        analysis[phase] = {}
        for step in representation[phase]:
            analysis[phase][step] = {}
            
            for ipo_key in ['inputs', 'outputs', 'methods']:
                if not any(tutorial.get(ipo_key) for tutorial in representation[phase][step].values()):
                    continue
                
                # Collect unique information per tutorial
                unique_info = {}
                all_info = set()
                for url, tutorial_data in representation[phase][step].items():
                    if ipo_key in tutorial_data:
                        info_set = set(tutorial_data[ipo_key])
                        unique_info[url] = info_set
                        all_info.update(info_set)
                
                # Find complementary and conflicting pairs
                complementary_pairs = []
                conflicting_pairs = []
                urls = list(unique_info.keys())
                
                for i, url1 in enumerate(urls):
                    for url2 in urls[i+1:]:
                        info1, info2 = unique_info[url1], unique_info[url2]
                        
                        # Check for complementary information
                        if info1 - info2 and info2 - info1:
                            complementary_pairs.append((url1, url2))
                        
                        # Check for potential conflicts
                        # This is a simplified check - you might want to use more sophisticated conflict detection
                        common_elements = info1 & info2
                        if common_elements and len(info1) != len(info2):
                            conflicting_pairs.append((url1, url2))
                
                # Calculate completeness scores
                completeness_scores = {
                    url: len(info) / len(all_info)
                    for url, info in unique_info.items()
                }
                
                # Calculate distribution statistics
                distribution_stats = {
                    'total_unique_elements': len(all_info),
                    'avg_elements_per_tutorial': np.mean([len(info) for info in unique_info.values()]),
                    'coverage_overlap': len(complementary_pairs) / (len(urls) * (len(urls) - 1) / 2) if len(urls) > 1 else 0
                }
                
                analysis[phase][step][ipo_key] = {
                    'unique_info': {url: list(info) for url, info in unique_info.items()},
                    'complementary_pairs': complementary_pairs,
                    'conflicting_pairs': conflicting_pairs,
                    'completeness_scores': completeness_scores,
                    'distribution_stats': distribution_stats
                }
    
    return analysis

def generate_summary_metrics(ipo_analysis: Dict, fragmentation_analysis: Dict) -> Dict:
    """
    Generate high-level summary metrics from both analyses.
    
    Returns:
    {
        'overall_diversity': float,  # Average diversity score across all steps
        'most_diverse_steps': List[str],  # Steps with highest IPO combination diversity
        'information_distribution': Dict,  # Statistics about information distribution
        'tutorial_complementarity': Dict,  # Metrics about tutorial relationships
    }
    """
    # Calculate overall diversity
    diversity_scores = []
    for phase in ipo_analysis:
        for step in ipo_analysis[phase]:
            diversity_scores.append(ipo_analysis[phase][step]['diversity_score'])
    
    overall_diversity = np.mean(diversity_scores) if diversity_scores else 0
    
    # Find most diverse steps
    step_diversity = []
    for phase in ipo_analysis:
        for step in ipo_analysis[phase]:
            step_diversity.append((f"{phase}: {step}", ipo_analysis[phase][step]['diversity_score']))
    
    most_diverse_steps = sorted(step_diversity, key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate information distribution metrics
    total_complementary_pairs = 0
    total_conflicting_pairs = 0
    completeness_scores = []
    
    for phase in fragmentation_analysis:
        for step in fragmentation_analysis[phase]:
            for ipo_key in fragmentation_analysis[phase][step]:
                analysis = fragmentation_analysis[phase][step][ipo_key]
                total_complementary_pairs += len(analysis['complementary_pairs'])
                total_conflicting_pairs += len(analysis['conflicting_pairs'])
                completeness_scores.extend(analysis['completeness_scores'].values())
    
    information_distribution = {
        'avg_tutorial_completeness': np.mean(completeness_scores) if completeness_scores else 0,
        'total_complementary_pairs': total_complementary_pairs,
        'total_conflicting_pairs': total_conflicting_pairs
    }
    
    return {
        'overall_diversity': overall_diversity,
        'most_diverse_steps': most_diverse_steps,
        'information_distribution': information_distribution,
        'tutorial_complementarity': {
            'complementary_ratio': total_complementary_pairs / (total_complementary_pairs + total_conflicting_pairs) if (total_complementary_pairs + total_conflicting_pairs) > 0 else 0
        }
    } 