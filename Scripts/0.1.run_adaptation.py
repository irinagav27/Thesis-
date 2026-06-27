"""
This code is a master scripts that connects the files of the four different strategies.
"""

import os
import sys
import subprocess

print("=" * 70)
print("Choose which strategy to run:")
print("=" * 70)

# Define available scripts
scripts = {
    "1": {
        "name": "Lexical Simplification",
        "file": "Scripts/1.lexical_adaptation.py",
        "description": "Replace complex vocabulary with simpler words\n            (Preserves sentence structure & grammar)",
        "output": "Data/adapted/lexical_simplified.csv",
    },
    "2": {
        "name": "Persona 1 Adaptation (8th-Grade English)",
        "file": "Scripts/2.persona_1_adaptation.py",
        "description": "Simplify for 13-15 year old readers\n            (Age: ~13-15, No art knowledge, Native English)",
        "output": "Data/adapted/persona1_adapted.csv",
    },
    "3": {
        "name": "Persona 2 Adaptation (Non-Native Expert)",
        "file": "Scripts/3.persona_2_adaptation.py",
        "description": "Simplify for art experts with non-native English\n            (English: B2-C1, Deep art knowledge, Non-native)",
        "output": "Data/adapted/persona2_adapted.csv",
    },
    "4": {
        "name": "Persona 3 Adaptation (Native Non-Expert)",
        "file": "Scripts/4.persona_3_adaptation.py",
        "description": "Simplify for general adult museum visitors\n            (English: Native, No art knowledge, College-educated)",
        "output": "Data/adapted/persona3_adapted.csv",
    },
    "all": {
        "name": "Run ALL Adaptations (1-4)",
        "file": None,
        "description": "Run lexical + all 3 personas in sequence",
        "output": "All above",
    }
}

# Display menu
print("\nAvailable Adaptation Scripts:\n")

for key, script in scripts.items():
    if key != "all":
        print(f"  [{key}] {script['name']}")
        print(f"      {script['description']}")
        print(f"      Output: {script['output']}")

print(f"  [all] Run ALL Adaptations (1-4 sequentially)")
print(f"      Runs all scripts in order")

print(f"  [quit/q] Exit without running anything\n")

# Get user choice
print("=" * 70)
while True:
    choice = input("Choose script to run (1-4, all, or quit): ").strip().lower()
    
    if choice in ["quit", "q", "exit"]:
        print("\nExiting without running anything.")
        sys.exit(0)
    
    if choice not in scripts.keys():
        print(" Invalid choice. Please enter 1, 2, 3, 4, all, or quit.")
        continue
    
    break

print("=" * 70)

# Run selected script(s)
if choice == "all":
    print("\nRunning ALL adaptations in sequence...\n")
    
    scripts_to_run = ["1", "2", "3", "4"]
    
    for script_num in scripts_to_run:
        script_info = scripts[script_num]
        script_file = script_info["file"]
        
        print(f"\n{'='*70}")
        print(f"Running Script {script_num}: {script_info['name']}")
        print(f"{'='*70}\n")
        
        # Check if file exists
        if not os.path.exists(script_file):
            print(f" ERROR: {script_file} not found!")
            print(f"  Please make sure the file exists in the scripts/ directory.")
            continue
        
        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, script_file],
                check=False,
                capture_output=False
            )
            
            if result.returncode == 0:
                print(f"\n Script {script_num} completed successfully!")
            else:
                print(f"\n Script {script_num} encountered an error (code {result.returncode})")
                print(f"  Continuing to next script...")
        
        except Exception as e:
            print(f"\n Error running script: {e}")
            print(f"  Continuing to next script...")
    
else:
    # Run single script
    script_info = scripts[choice]
    script_file = script_info["file"]
    
    print(f"\nRunning Script {choice}: {script_info['name']}\n")
    print(f"Description: {script_info['description']}")
    print(f"Output file: {script_info['output']}")
    
    print("=" * 70)
    
    # Run the script
    print(f"Starting execution...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_file],
            check=False
        )
        
        print("\n" + "=" * 70)
        
        if result.returncode == 0:
            print(f" SCRIPT {choice} COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\nOutput saved to: {script_info['output']}")
        else:
            print(f" SCRIPT {choice} ENCOUNTERED AN ERROR!")
            print(f"   Return code: {result.returncode}")
            print("=" * 70)
            print("\nPlease check the error messages above.")
            print("Common issues:")
            print("  - API key not set in config/api_keys.env")
            print("  - Dataset not found at Data/Original/Dataset_Curatorial_Text.xlsx")
            print("  - Prompts not found in Prompts")
            print("  - Missing Python packages (pip install -r requirements.txt)")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(" SCRIPT INTERRUPTED BY USER")
        print("=" * 70)
        print("\nTo resume, run: python run_adaptation.py")
    
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("=" * 70)
        sys.exit(1)

print("\n")