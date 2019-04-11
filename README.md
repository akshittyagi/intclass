# Fast Intent Classification

## Model Performance

### Facebook Semantic 
| Model         | Accuracy/F1(micr)| F1(Macro)        |
|:-------------:|:----------------:|:----------------:|
| 3Layer        | 88.5             | 0.48             |
| 3LayerBN      | 89.6             | 0.55             |
| S-LSTM        | 92.8             | 0.65             |
| S-LSTM-BN     | 93.2             | 0.66             |

| Model         | Exit Point(%)    | Accuracy         |
|:-------------:|:----------------:|:----------------:|
| 3LayerBN      | 3: 70.2          | 89.27            |
|               | 1: 27.8          |                  |
|               | 2: 1.92          |                  |
| 3LayerBN      | 3: 67.9          | 85.03            |
|               | 1: 29.3          |                  |
|               | 2: 2.6           |                  |
| 3LayerBN      | 3: 70.80         | 87.84            |
|               | 1: 27.37         |                  |
|               | 2: 1.80          |                  |
