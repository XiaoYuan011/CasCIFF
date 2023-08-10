import numpy as np

def Old_Evaluate(prediction, truth):
    prediction = np.array(prediction)
    truth = np.array(truth)
    predictions = np.array([label+1 for label in prediction])
    test_label = np.array([label+1 for label in truth])
    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(test_label)))
    report_mape = np.mean(np.abs(np.log2(predictions + 1) - np.log2(test_label + 1)) / np.log2(test_label + 2))
    report_mae = np.mean(np.abs(np.log2(predictions) - np.log2(test_label)))
    report_mrse = np.mean(
        np.square(np.abs(np.log2(predictions + 1) - np.log2(test_label + 1)) / np.log2(test_label + 2)))
    print('Old Metrics TestDateSet MSLE: {:.4f}, MAPE: {:.4f}, MAE: {:.4f}, MRSE: {:.4f}'.format(report_msle, report_mape, report_mae,
                                                                              report_mrse))
    return 'Old Metrics TestDateSet MSLE: {:.4f}, MAPE: {:.4f}, MAE: {:.4f}, MRSE: {:.4f}'.format(report_msle, report_mape, report_mae,
                                                                              report_mrse)
def New_Evaluate(prediction, truth):
    prediction = np.array(prediction)
    truth = np.array(truth)
    predictions = np.array([1 if label < 1 else label for label in prediction])
    test_label = np.array([1 if label < 1 else label for label in truth])
    report_msle = np.mean(np.square(np.log2(predictions) - np.log2(test_label)))
    report_mape = np.mean(np.abs(np.log2(predictions + 1) - np.log2(test_label + 1)) / np.log2(test_label + 2))
    report_mae = np.mean(np.abs(np.log2(predictions) - np.log2(test_label)))
    report_mrse = np.mean(
        np.square(np.abs(np.log2(predictions + 1) - np.log2(test_label + 1)) / np.log2(test_label + 2)))
    print('New Metrics TestDateSet MSLE: {:.4f}, MAPE: {:.4f}, MAE: {:.4f}, MRSE: {:.4f}'.format(report_msle, report_mape,
                                                                                             report_mae,
                                                                                             report_mrse))
    return 'New Metrics TestDateSet MSLE: {:.4f}, MAPE: {:.4f}, MAE: {:.4f}, MRSE: {:.4f}'.format(report_msle, report_mape,
                                                                                              report_mae,
                                                                                              report_mrse)