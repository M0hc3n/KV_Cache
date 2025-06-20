import torch
import time
import numpy as np

from model import ModelArgs as args, Transformer

def demonstrate_kv_caching():
    """
    Demonstrates KV-caching functionality with timing comparisons
    """
    print("=" * 60)
    print("KV-CACHE DEMONSTRATION")
    print("=" * 60)

    print("Model Configuration:")
    print(f"  - Dimensions: {args.dim}")
    print(f"  - Layers: {args.n_layers}")
    print(f"  - Attention Heads: {args.n_heads}")
    print(f"  - KV Heads: {args.n_kv_heads}")
    print(f"  - Vocabulary Size: {args.vocab_size}")
    print(f"  - Device: {args.device}")
    print()

    # Initialize model and tokenizer
    model = Transformer(args()).to(args.device)

    # Put model in evaluation mode
    model.eval()

    print("Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_single_token_generation():
    """Test generating tokens one by one (simulating KV-cache usage)"""
    print("=" * 40)
    print("TESTING SINGLE TOKEN GENERATION")
    print("=" * 40)

    model = Transformer(args()).to(args.device)
    model.eval()

    # Initial prompt
    prompt_tokens = [0, 10, 25, 42]  # Simulated token sequence

    print(f"Initial prompt tokens: {prompt_tokens}")
    print(f"Device: {args.device}")
    print()

    # Process initial prompt
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)

    with torch.no_grad():
        print("Processing initial prompt...")
        start_time = time.time()

        # Process the initial prompt
        for i in range(len(prompt_tokens)):
            token_input = prompt_tensor[:, i : i + 1]  # Single token
            logits = model(token_input, start_pos=i)
            print(
                f"  Token {i + 1}/{len(prompt_tokens)}: {prompt_tokens[i]} -> logits shape: {logits.shape}"
            )

        prompt_time = time.time() - start_time
        print(f"Prompt processing time: {prompt_time:.4f}s")
        print()

        # Generate new tokens
        print("Generating new tokens...")
        generated_tokens = prompt_tokens.copy()
        generation_times = []

        for step in range(5):  # Generate 5 new tokens
            start_time = time.time()

            # Use the last token as input
            current_token = torch.tensor(
                [[generated_tokens[-1]]], dtype=torch.long, device=args.device
            )
            start_pos = len(generated_tokens) - 1

            # Forward pass
            logits = model(current_token, start_pos=start_pos)

            # Sample next token (using simple argmax for deterministic results)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated_tokens.append(next_token)

            step_time = time.time() - start_time
            generation_times.append(step_time)

            print(
                f"  Step {step + 1}: Generated token {next_token} (time: {step_time:.4f}s)"
            )

        avg_generation_time = np.mean(generation_times)
        print(f"\nAverage generation time per token: {avg_generation_time:.4f}s")
        print(f"Final sequence: {generated_tokens}")


def demonstrate_cache_inspection():
    """Inspect the KV cache to show it's working"""
    print("\n" + "=" * 40)
    print("INSPECTING KV CACHE")
    print("=" * 40)

    model = Transformer(args()).to(args.device)
    model.eval()

    # Get the first attention layer for inspection
    attention_layer = model.layers[0].attention

    print("Cache shapes:")
    print(f"  Keys cache: {attention_layer.cache_keys.shape}")
    print(f"  Values cache: {attention_layer.cache_values.shape}")
    print()

    # Process a sequence token by token
    tokens = [5, 10, 15, 20]

    with torch.no_grad():
        for i, token in enumerate(tokens):
            print(f"Processing token {i + 1}: {token}")

            # Check cache before processing
            keys_before = attention_layer.cache_keys[
                0, i, 0, :5
            ].clone()  # First 5 dims
            values_before = attention_layer.cache_values[0, i, 0, :5].clone()

            # Process token
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=args.device)
            _ = model(token_tensor, start_pos=i)

            # Check cache after processing
            keys_after = attention_layer.cache_keys[0, i, 0, :5]
            values_after = attention_layer.cache_values[0, i, 0, :5]

            print(f"  Keys cache at pos {i}:")
            print(f"    Before: {keys_before.cpu().numpy()}")
            print(f"    After:  {keys_after.cpu().numpy()}")
            print(f"    Changed: {not torch.allclose(keys_before, keys_after)}")
            print()


def benchmark_with_without_cache():
    """Compare performance with and without KV caching (conceptual)"""
    print("=" * 40)
    print("KV-CACHE PERFORMANCE BENEFITS")
    print("=" * 40)

    sequence_lengths = [10, 50, 100, 500, 1000]

    print("   Theoretical speedup for different sequence lengths:")
    print("   (comparing cached vs non-cached token generation)")
    print(
        f"   {'Seq Length':<12} {'Without Cache':<15} {'With Cache':<12} {'Speedup':<10}"
    )
    print("   " + "-" * 50)

    for seq_len in sequence_lengths:
        without_cache_ops = seq_len * seq_len  # O(n²) for attention
        with_cache_ops = seq_len  # O(n) with caching
        speedup = without_cache_ops / with_cache_ops
        print(
            f"   {seq_len:<12} {without_cache_ops:<15} {with_cache_ops:<12} {speedup:.1f}x"
        )


def main():
    """Main demonstration function"""
    print("🚀 Starting KV-Cache Demonstration")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run demonstrations
        demonstrate_kv_caching()
        test_single_token_generation()
        demonstrate_cache_inspection()
        benchmark_with_without_cache()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or device issues.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
