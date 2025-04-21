import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Bayesian Network Examples')
    parser.add_argument('--example', type=str, 
                        choices=['sprinkler', 'asia', 'high_utility', 'all'], 
                        default='all', help='Example to run (default: all)')
    parser.add_argument('--no-plots', action='store_true', 
                        help='Skip visualization plots (for high_utility example)')
    
    args = parser.parse_args()
    
    if args.example == 'sprinkler' or args.example == 'all':
        print("=" * 80)
        print("Running Sprinkler Network Example")
        print("=" * 80)
        from src.example import main as sprinkler_main
        sprinkler_main()
        print("\n")
    
    if args.example == 'asia' or args.example == 'all':
        print("=" * 80)
        print("Running Asia Network Example with Variable Elimination")
        print("=" * 80)
        from src.variable_elimination_example import main as asia_main
        asia_main()
        print("\n")
    
    if args.example == 'high_utility' or args.example == 'all':
        print("=" * 80)
        print("Running Top-K High Utility Bayesian Networks Example")
        print("=" * 80)
        if args.no_plots:
            sys.argv.append('--no-plots')
        from src.top_k_high_utility_example import main as high_utility_main
        high_utility_main()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 