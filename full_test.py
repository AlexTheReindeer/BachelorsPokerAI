import csv
import os
import subprocess
import sys
from datetime import datetime

def install_required_packages():
    """Install required packages if they're not already installed"""
    try:
        import pypokerengine
        print("pypokerengine is already installed.")
    except ImportError:
        print("Installing required packages...")
        try:
            # Try installing with pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypokerengine"])
            print("Packages installed successfully!")
            
            # Verify installation
            import pypokerengine
            print("Package verification successful!")
        except Exception as e:
            print(f"Failed to install pypokerengine: {str(e)}")
            print("Please install pypokerengine manually using:")
            print("pip install pypokerengine")
            sys.exit(1)

# Install required packages before proceeding
install_required_packages()

class FullTest:
    def __init__(self, output_file='full_test_results.csv'):
        self.output_file = output_file
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Session',
                    'Training Win Rate',
                    'Training Total Reward',
                    'Training Avg Reward',
                    'Test Win Rate',
                    'Test Total Reward',
                    'Test Avg Reward'
                ])
    
    def _save_results(self, session, train_results, test_results):
        """Save results to CSV file"""
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                session,
                train_results['win_rate'],
                train_results['total_reward'],
                train_results['avg_reward'],
                test_results['win_rate'],
                test_results['total_reward'],
                test_results['avg_reward']
            ])
    
    def _parse_output(self, output):
        """Parse the output from trainer.py to extract metrics"""
        print("Raw output for debugging:")
        print(output)
        print("-" * 50)
        
        lines = output.split('\n')
        win_rate = 0
        total_reward = 0
        avg_reward = 0
        
        for line in lines:
            if 'Win Rate:' in line:
                try:
                    win_rate = float(line.split(':')[1].strip().replace('%', '')) / 100
                except (ValueError, IndexError):
                    print(f"Error parsing win rate from line: {line}")
            elif 'Total Reward:' in line:
                try:
                    total_reward = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    print(f"Error parsing total reward from line: {line}")
            elif 'Average Reward:' in line:
                try:
                    avg_reward = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    print(f"Error parsing average reward from line: {line}")
        
        return {
            'win_rate': win_rate,
            'total_reward': total_reward,
            'avg_reward': avg_reward
        }
    
    def run_training(self):
        """Run training mode and return results"""
        print("Running training command...")
        result = subprocess.run(
            [sys.executable, 'trainer.py', '--mode', 'train', '--episodes', '3000'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error in training command: {result.stderr}")
            return None
        return self._parse_output(result.stdout)
    
    def run_testing(self):
        """Run testing mode and return results"""
        print("Running testing command...")
        result = subprocess.run(
            [sys.executable, 'trainer.py', '--mode', 'test', '--episodes', '1000'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error in testing command: {result.stderr}")
            return None
        return self._parse_output(result.stdout)
    
    def run_full_test(self, total_sessions=14):
        """Run the full test sequence"""
        for session in range(1, total_sessions + 1):
            print(f"\nStarting Session {session}/{total_sessions}")
            
            # Run training
            print("Running training phase...")
            train_results = self.run_training()
            if train_results is None:
                print("Training failed, stopping test sequence.")
                return
            
            # Run testing
            print("Running testing phase...")
            test_results = self.run_testing()
            if test_results is None:
                print("Testing failed, stopping test sequence.")
                return
            
            # Save results
            self._save_results(session, train_results, test_results)
            
            # Print session summary
            print(f"\nSession {session} Summary:")
            print(f"Training Win Rate: {train_results['win_rate']:.2%}")
            print(f"Training Total Reward: {train_results['total_reward']:.2f}")
            print(f"Training Avg Reward: {train_results['avg_reward']:.2f}")
            print(f"Test Win Rate: {test_results['win_rate']:.2%}")
            print(f"Test Total Reward: {test_results['total_reward']:.2f}")
            print(f"Test Avg Reward: {test_results['avg_reward']:.2f}")
            print("-" * 50)

if __name__ == "__main__":
    full_test = FullTest()
    full_test.run_full_test(14) 