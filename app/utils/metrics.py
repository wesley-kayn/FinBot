import time
import json
from datetime import datetime
from typing import Dict, Any, List

class MetricsCollector:
    """Collect and track metrics for the Finbot LLM application"""
    
    def __init__(self):
        self.session_start = time.time()
        self.query_count = 0
        self.response_times = []
        self.errors = []
        self.jailbreak_attempts = 0
        self.out_of_domain_queries = 0
        
    def record_query(self, response_time: float, is_jailbreak: bool = False, is_out_of_domain: bool = False):
        """Record a query and its metrics"""
        self.query_count += 1
        self.response_times.append(response_time)
        
        if is_jailbreak:
            self.jailbreak_attempts += 1
        if is_out_of_domain:
            self.out_of_domain_queries += 1
    
    def record_error(self, error: str):
        """Record an error"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        session_duration = time.time() - self.session_start
        
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'session_duration': session_duration,
            'total_queries': self.query_count,
            'average_response_time': avg_response_time,
            'jailbreak_attempts': self.jailbreak_attempts,
            'out_of_domain_queries': self.out_of_domain_queries,
            'error_count': len(self.errors),
            'session_start': datetime.fromtimestamp(self.session_start).isoformat()
        }
    
    def print_session_stats(self):
        """Print session statistics to console"""
        stats = self.get_session_stats()
        
        print("\n===== FINBOT LLM SESSION STATISTICS =====")
        print(f"Session Duration: {stats['session_duration']:.2f} seconds")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Average Response Time: {stats['average_response_time']:.3f} seconds")
        print(f"Jailbreak Attempts: {stats['jailbreak_attempts']}")
        print(f"Out-of-Domain Queries: {stats['out_of_domain_queries']}")
        print(f"Errors: {stats['error_count']}")
        print(f"Session Start: {stats['session_start']}")
        print("==========================================\n")
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finbot_metrics_{timestamp}.json"
        
        metrics_data = {
            'session_stats': self.get_session_stats(),
            'response_times': self.response_times,
            'errors': self.errors
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return filename

# Global metrics collector instance
metrics_collector = MetricsCollector()