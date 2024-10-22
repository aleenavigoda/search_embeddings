import psutil
import platform
import os
import subprocess
import multiprocessing
import json
from datetime import datetime

class SystemDiagnostics:
    def __init__(self):
        self.specs = {}
        
    def get_mac_chip_info(self):
        """Get Mac chip information"""
        try:
            # Try to get chip information on Mac
            cmd = "sysctl -n machdep.cpu.brand_string"
            chip_info = subprocess.check_output(cmd.split()).decode().strip()
            
            # Additional Mac-specific info
            model_cmd = "sysctl hw.model"
            model = subprocess.check_output(model_cmd.split()).decode().strip()
            
            return {
                'chip_type': chip_info,
                'model': model
            }
        except:
            return None
    
    def get_memory_info(self):
        """Get detailed memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_memory_gb': round(memory.total / (1024**3), 2),
            'available_memory_gb': round(memory.available / (1024**3), 2),
            'used_memory_gb': round(memory.used / (1024**3), 2),
            'memory_percent': memory.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2)
        }
    
    def get_cpu_info(self):
        """Get CPU information"""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'total_cores': psutil.cpu_count(logical=True),
            'current_frequency': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None,
            'min_frequency': psutil.cpu_freq().min if hasattr(psutil.cpu_freq(), 'min') else None,
            'max_frequency': psutil.cpu_freq().max if hasattr(psutil.cpu_freq(), 'max') else None
        }
    
    def get_disk_info(self):
        """Get disk information"""
        disk = psutil.disk_usage('/')
        return {
            'total_disk_gb': round(disk.total / (1024**3), 2),
            'free_disk_gb': round(disk.free / (1024**3), 2),
            'used_disk_gb': round(disk.used / (1024**3), 2),
            'disk_percent': disk.percent
        }
    
    def calculate_safe_parameters(self):
        """Calculate safe parameters for data processing"""
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        # Safe parameters calculation
        safe_params = {
            'max_workers': min(multiprocessing.cpu_count(), 8),  # Conservative worker count
            'max_memory_usage_gb': round(total_memory_gb * 0.4, 2),  # Use max 40% of total memory
            'safe_batch_size': int((total_memory_gb * 0.1) * 1000),  # 10% of memory per batch
            'recommended_target_dataset_size_gb': round(min(total_memory_gb * 0.3, 10), 2),  # 30% of RAM or 10GB, whichever is smaller
        }
        
        return safe_params
    
    def run_diagnostics(self):
        """Run full system diagnostics"""
        # System information
        self.specs['system'] = {
            'os': platform.system(),
            'os_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # Mac-specific information
        if platform.system() == 'Darwin':
            self.specs['mac_info'] = self.get_mac_chip_info()
        
        # Hardware information
        self.specs['memory'] = self.get_memory_info()
        self.specs['cpu'] = self.get_cpu_info()
        self.specs['disk'] = self.get_disk_info()
        
        # Safe parameters
        self.specs['safe_parameters'] = self.calculate_safe_parameters()
        
        return self.specs
    
    def save_diagnostics(self):
        """Save diagnostics to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'system_diagnostics_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.specs, f, indent=4)
        
        return filename

def main():
    diagnostics = SystemDiagnostics()
    specs = diagnostics.run_diagnostics()
    
    print("\n=== System Diagnostics ===\n")
    
    # Print system info
    if specs.get('mac_info'):
        print("Mac System Information:")
        print(f"Chip: {specs['mac_info'].get('chip_type', 'Unknown')}")
        print(f"Model: {specs['mac_info'].get('model', 'Unknown')}")
    
    # Print memory info
    print("\nMemory Information:")
    print(f"Total Memory: {specs['memory']['total_memory_gb']} GB")
    print(f"Available Memory: {specs['memory']['available_memory_gb']} GB")
    print(f"Memory Usage: {specs['memory']['memory_percent']}%")
    
    # Print CPU info
    print("\nCPU Information:")
    print(f"Physical Cores: {specs['cpu']['physical_cores']}")
    print(f"Total Cores: {specs['cpu']['total_cores']}")
    
    # Print safe parameters
    print("\nRecommended Safe Parameters:")
    safe_params = specs['safe_parameters']
    print(f"Max Workers: {safe_params['max_workers']}")
    print(f"Max Memory Usage: {safe_params['max_memory_usage_gb']} GB")
    print(f"Safe Batch Size: {safe_params['safe_batch_size']} samples")
    print(f"Recommended Dataset Size: {safe_params['recommended_target_dataset_size_gb']} GB")
    
    # Save diagnostics
    filename = diagnostics.save_diagnostics()
    print(f"\nFull diagnostics saved to: {filename}")

if __name__ == "__main__":
    main()