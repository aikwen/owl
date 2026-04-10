from src.owl.toolkits.common.fmt import format_metrics_table

if __name__ == "__main__":
    mock_metrics = {
        "val_mask_01": {"auc": 0.9921, "f1_score": 0.9810},
        "test_set_hard": {"auc": 0.9540, "f1_score": 0.9105}
    }

    table_string = format_metrics_table(mock_metrics, current_epoch=1)

    print(table_string)