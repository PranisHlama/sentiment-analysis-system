# For model comparison reporting so Logistic Regression, LSTM, and BERT results can be reviewed side by side.

import pandas as pd

def build_model_comparison(results):
    rows = []
    for model_name, result in results.items():
        rows.append({
            "model": model_name,
            "accuracy": result.get("accuracy")
        })

    return pd.DataFrame(rows).sort_values(
        by="accuracy",
        ascending=False
    )

def save_model_comparison(results, output_path):
    comparison = build_model_comparison(results)
    comparison.to_csv(output_path, index=False)

    print("\n Model Comparison")
    print(comparison.to_string(index=False))

    return comparison

