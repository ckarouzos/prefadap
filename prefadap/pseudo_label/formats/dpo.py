"""DPO format handler for preference datasets."""

from __future__ import annotations

from typing import Any, Dict


class DPOFormatter:
    """Formatter for DPO/ORPO preference datasets."""
    
    @staticmethod
    def format_preference(
        prompt: str,
        chosen: str, 
        rejected: str,
        weight: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Format a preference pair for DPO/ORPO training.
        
        Parameters
        ----------
        prompt : str
            Input prompt/context
        chosen : str
            Preferred response
        rejected : str
            Rejected response  
        weight : float
            Sample weight (default: 1.0)
        **kwargs
            Additional fields to include
            
        Returns
        -------
        Dict[str, Any]
            DPO-formatted record
        """
        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "weight": weight,
        }
        
        # Add any additional fields
        record.update(kwargs)
        
        return record
    
    @staticmethod
    def format_multiple_preferences(
        prompt: str,
        candidates: list[tuple[str, float]],
        max_pairs: int = 5,
        **kwargs
    ) -> list[Dict[str, Any]]:
        """Format multiple candidates into preference pairs.
        
        Parameters
        ----------
        prompt : str
            Input prompt/context
        candidates : list[tuple[str, float]]
            List of (candidate_text, score) tuples, sorted by score descending
        max_pairs : int
            Maximum number of preference pairs to generate
        **kwargs
            Additional fields to include in each record
            
        Returns
        -------
        list[Dict[str, Any]]
            List of DPO-formatted records
        """
        if len(candidates) < 2:
            return []
        
        records = []
        pairs_generated = 0
        
        # Create pairs from ranked candidates
        for i in range(len(candidates) - 1):
            if pairs_generated >= max_pairs:
                break
                
            chosen_text, chosen_score = candidates[i]
            
            for j in range(i + 1, len(candidates)):
                if pairs_generated >= max_pairs:
                    break
                    
                rejected_text, rejected_score = candidates[j]
                
                # Only create pair if there's a meaningful score difference
                if chosen_score > rejected_score:
                    record = DPOFormatter.format_preference(
                        prompt=prompt,
                        chosen=chosen_text,
                        rejected=rejected_text,
                        weight=1.0,
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                        **kwargs
                    )
                    records.append(record)
                    pairs_generated += 1
        
        return records


__all__ = ["DPOFormatter"]