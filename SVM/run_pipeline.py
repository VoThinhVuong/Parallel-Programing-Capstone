"""
Complete SVM Pipeline: Train and Evaluate
Combines Steps 4 & 5 into a single script for convenience
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*70)
    print(" CIFAR-10 SVM Classification Pipeline")
    print(" Steps 4 & 5: Train and Evaluate")
    print("="*70)
    
    # Import and run training
    print("\n\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " STEP 4: TRAINING SVM ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    try:
        import train_svm
        train_svm.main()
    except Exception as e:
        print(f"\n❌ ERROR in training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Import and run evaluation
    print("\n\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " STEP 5: EVALUATING SVM ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    try:
        import evaluate_svm
        evaluate_svm.main()
    except Exception as e:
        print(f"\n❌ ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n\n" + "="*70)
    print(" ✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - svm_model.pkl          : Trained SVM model")
    print("  - evaluation_results.txt : Detailed metrics")
    print("  - confusion_matrix.png   : Visualization")
    print("="*70)


if __name__ == '__main__':
    main()
