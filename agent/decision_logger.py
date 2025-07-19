"""
Decision Logger for AutoSocialMedia Pipeline

Captures AI reasoning and decision-making process for debugging and improvement.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Decision:
    """Represents a single AI decision with reasoning."""
    component: str
    step: str
    decision: str
    reasoning: str
    alternatives_considered: List[str]
    confidence_score: float
    input_data: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any] = None

class DecisionLogger:
    """Logs AI decisions and reasoning throughout the pipeline."""
    
    def __init__(self, run_dir: str, debug_mode: bool = False):
        self.run_dir = run_dir
        self.debug_mode = debug_mode
        self.decisions = []
        self.component_decisions = {}
        self.current_component = None
        
        # Create decisions directory
        self.decisions_dir = os.path.join(run_dir, 'decisions')
        os.makedirs(self.decisions_dir, exist_ok=True)
    
    def start_component(self, component_name: str):
        """Start logging for a specific component."""
        self.current_component = component_name
        self.component_decisions[component_name] = []
        logging.info(f"Started decision logging for component: {component_name}")
    
    def log_decision(self, step: str, decision: str, reasoning: str, 
                    alternatives: List[str] = None, confidence: float = 1.0,
                    input_data: Dict = None, metadata: Dict = None):
        """Log a single decision with reasoning."""
        
        if not self.current_component:
            logging.warning("Decision logged without active component context")
            return
        
        decision_obj = Decision(
            component=self.current_component,
            step=step,
            decision=decision,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence_score=confidence,
            input_data=input_data or {},
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Add to both global and component-specific logs
        self.decisions.append(decision_obj)
        self.component_decisions[self.current_component].append(decision_obj)
        
        # Log to console if debug mode
        if self.debug_mode:
            logging.info(f"DECISION [{self.current_component}] {step}: {decision}")
            logging.info(f"   Reasoning: {reasoning}")
            if alternatives:
                logging.info(f"   Alternatives: {alternatives}")
            logging.info(f"   Confidence: {confidence:.2f}")
    
    def log_analysis_result(self, analysis_type: str, result: Any, reasoning: str,
                          scores: Dict[str, float] = None):
        """Log analysis results with detailed reasoning."""
        
        metadata = {
            "analysis_type": analysis_type,
            "result": result,
            "scores": scores or {}
        }
        
        self.log_decision(
            step=f"{analysis_type}_analysis",
            decision=str(result),
            reasoning=reasoning,
            confidence=max(scores.values()) if scores else 1.0,
            metadata=metadata
        )
    
    def log_fallback_decision(self, intended_action: str, fallback_action: str, 
                            reason: str, impact: str):
        """Log fallback decisions and their impact."""
        
        self.log_decision(
            step="fallback_decision",
            decision=f"Using fallback: {fallback_action}",
            reasoning=f"Intended: {intended_action}. Reason: {reason}. Impact: {impact}",
            alternatives=[intended_action],
            confidence=0.5,  # Lower confidence for fallbacks
            metadata={
                "is_fallback": True,
                "intended_action": intended_action,
                "impact_assessment": impact
            }
        )
    
    def finish_component(self):
        """Finish logging for current component and save component log."""
        if not self.current_component:
            return
        
        # Save component-specific decision log
        component_log_path = os.path.join(
            self.decisions_dir, 
            f"{self.current_component}_decisions.json"
        )
        
        component_decisions = [asdict(d) for d in self.component_decisions[self.current_component]]
        
        with open(component_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "component": self.current_component,
                "total_decisions": len(component_decisions),
                "decisions": component_decisions,
                "summary": self._create_component_summary(self.current_component)
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(component_decisions)} decisions for {self.current_component}")
        self.current_component = None
    
    def _reset_current_component(self):
        """Reset current component without saving decision file (for components without decisions)."""
        if self.current_component:
            # Remove any decisions that might have been added
            if self.current_component in self.component_decisions:
                # Remove from global decisions list
                component_decisions = self.component_decisions[self.current_component]
                for decision in component_decisions:
                    if decision in self.decisions:
                        self.decisions.remove(decision)
                # Clear component decisions
                del self.component_decisions[self.current_component]
            self.current_component = None
    
    def save_master_log(self):
        """Save comprehensive decision log at end of pipeline."""
        
        master_log_path = os.path.join(self.decisions_dir, 'master_decisions.json')
        
        all_decisions = [asdict(d) for d in self.decisions]
        
        master_log = {
            "pipeline_run": {
                "timestamp": datetime.now().isoformat(),
                "total_decisions": len(all_decisions),
                "components_analyzed": list(self.component_decisions.keys())
            },
            "summary": self._create_master_summary(),
            "decisions_by_component": {
                comp: [asdict(d) for d in decisions] 
                for comp, decisions in self.component_decisions.items()
            },
            "all_decisions_chronological": all_decisions
        }
        
        with open(master_log_path, 'w', encoding='utf-8') as f:
            json.dump(master_log, f, indent=2, ensure_ascii=False)
        
        # Also create a human-readable summary
        self._create_readable_summary()
        
        logging.info(f"Saved master decision log with {len(all_decisions)} total decisions")
        return master_log_path
    
    def _create_component_summary(self, component: str) -> Dict[str, Any]:
        """Create summary for a specific component."""
        decisions = self.component_decisions[component]
        
        if not decisions:
            return {"message": "No decisions logged"}
        
        # Calculate confidence statistics
        confidences = [d.confidence_score for d in decisions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Count fallbacks
        fallback_count = sum(1 for d in decisions if d.metadata and d.metadata.get('is_fallback'))
        
        # Get key decisions
        key_decisions = [d.step for d in decisions if d.confidence_score >= 0.8]
        
        return {
            "total_decisions": len(decisions),
            "average_confidence": round(avg_confidence, 2),
            "fallback_decisions": fallback_count,
            "key_decisions": key_decisions,
            "decision_steps": list(set(d.step for d in decisions))
        }
    
    def _create_master_summary(self) -> Dict[str, Any]:
        """Create comprehensive pipeline summary."""
        
        if not self.decisions:
            return {"message": "No decisions logged"}
        
        # Overall statistics
        total_decisions = len(self.decisions)
        avg_confidence = sum(d.confidence_score for d in self.decisions) / total_decisions if total_decisions > 0 else 0.0
        
        # Component-wise breakdown
        component_stats = {}
        for comp, decisions in self.component_decisions.items():
            if decisions:
                component_stats[comp] = {
                    "decisions": len(decisions),
                    "avg_confidence": round(sum(d.confidence_score for d in decisions) / len(decisions), 2) if decisions else 0.0,
                    "fallbacks": sum(1 for d in decisions if d.metadata and d.metadata.get('is_fallback'))
                }
        
        # Quality indicators
        quality_indicators = {
            "high_confidence_decisions": sum(1 for d in self.decisions if d.confidence_score >= 0.8),
            "low_confidence_decisions": sum(1 for d in self.decisions if d.confidence_score < 0.5),
            "total_fallbacks": sum(1 for d in self.decisions if d.metadata and d.metadata.get('is_fallback'))
        }
        
        return {
            "total_decisions": total_decisions,
            "average_confidence": round(avg_confidence, 2),
            "component_breakdown": component_stats,
            "quality_indicators": quality_indicators
        }
    
    def _create_readable_summary(self):
        """Create human-readable summary file."""
        
        summary_path = os.path.join(self.decisions_dir, 'decision_summary.md')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# AutoSocialMedia Decision Log Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall stats
            f.write("## Pipeline Overview\n\n")
            f.write(f"- **Total Decisions:** {len(self.decisions)}\n")
            f.write(f"- **Components Analyzed:** {len(self.component_decisions)}\n")
            avg_conf = sum(d.confidence_score for d in self.decisions) / len(self.decisions) if self.decisions else 0.0
            f.write(f"- **Average Confidence:** {avg_conf:.2f}\n\n")
            
            # Component breakdown
            f.write("## Component Analysis\n\n")
            for comp, decisions in self.component_decisions.items():
                if not decisions:
                    continue
                    
                f.write(f"### {comp.title()}\n\n")
                
                # Key decisions for this component
                key_decisions = [d for d in decisions if d.confidence_score >= 0.8]
                if key_decisions:
                    f.write("**Key Decisions:**\n")
                    for d in key_decisions[:5]:  # Top 5
                        f.write(f"- **{d.step}:** {d.decision}\n")
                        f.write(f"  - *Reasoning:* {d.reasoning}\n")
                        f.write(f"  - *Confidence:* {d.confidence_score:.2f}\n\n")
                
                # Fallbacks for this component
                fallbacks = [d for d in decisions if d.metadata and d.metadata.get('is_fallback')]
                if fallbacks:
                    f.write("**Fallback Decisions:**\n")
                    for d in fallbacks:
                        f.write(f"- **{d.step}:** {d.decision}\n")
                        f.write(f"  - *Reason:* {d.reasoning}\n\n")
            
            f.write("---\n\n")
            f.write("*This summary was automatically generated to help improve video quality.*\n")

# Global instance for easy access
_decision_logger: Optional[DecisionLogger] = None

def init_decision_logger(run_dir: str, debug_mode: bool = False):
    """Initialize global decision logger."""
    global _decision_logger
    _decision_logger = DecisionLogger(run_dir, debug_mode)

def get_decision_logger() -> DecisionLogger:
    """Get the global decision logger instance."""
    if _decision_logger is None:
        raise RuntimeError("Decision logger not initialized. Call init_decision_logger() first.")
    return _decision_logger

def log_decision(step: str, decision: str, reasoning: str, **kwargs):
    """Convenient function to log decisions."""
    if _decision_logger:
        _decision_logger.log_decision(step, decision, reasoning, **kwargs) 