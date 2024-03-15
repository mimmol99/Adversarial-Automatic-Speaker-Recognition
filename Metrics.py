#@title Metrics

import numpy as np
from numpy import linalg as LA
from pesq import pesq
from pystoi import stoi

Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of ASV system falsely rejecting target speaker
    'Cfa': 10,  # Cost of ASV system falsely accepting nontarget speaker
    'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
    'Cfa_asv':10,  # Cost of ASV system falsely accepting nontarget speaker
    'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
    'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    'Cfa_spoof': 10,#added# # Cost of ASV system falsely accepting a spoof attempt
}



###AUC###

import torch
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def compute_auc(model,data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)[:, 1]  # Assuming binary classification
            predictions.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
    return roc_auc

def plot_auc(model, data_loader):
    roc_auc = compute_auc(model,data_loader)

    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

###SCORES###

def compute_scores_loader(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    target_scores = []
    nontarget_scores = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            inputs.to('cpu')
            labels.to('cpu')

            # Assuming outputs are probabilities for class 1 ('authorized')
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            labels = labels.cpu().numpy()  # Move labels to CPU and convert to numpy

            target_scores.extend(scores[labels == 1])
            nontarget_scores.extend(scores[labels == 0])
    return np.array(target_scores), np.array(nontarget_scores)


def compute_scores_samples(model, samples, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    target_scores = []
    nontarget_scores = []

    with torch.no_grad():
        inputs = torch.tensor(samples).to(device)
        outputs = model(inputs)
        scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Move to CPU and convert to numpy

        # Ensure labels are on the CPU before converting to numpy
        labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

        # Separate scores into target and non-target based on labels
        target_scores = scores[labels == 1]
        nontarget_scores = scores[labels == 0]

    return np.array(target_scores), np.array(nontarget_scores)


###EER###

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_eer_samples(model, samples, labels):
    target_scores, nontarget_scores = compute_scores_samples(model, samples, labels)
    eer, threshold = compute_eer(target_scores, nontarget_scores)
    return eer, threshold



def compute_eer_loader(model, loader):
    target_scores, nontarget_scores = compute_scores_loader(model, loader)
    eer, threshold = compute_eer(target_scores, nontarget_scores)
    return eer, threshold



###DET CURVE###
def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_det_curve_loader(model,loader):
    target_scores, nontarget_scores = compute_scores_loader(model, loader)
    frr, far, thresholds= compute_det_curve(target_scores, nontarget_scores)
    return frr, far, thresholds

def compute_det_curve_samples(model,samples, labels):
    target_scores, nontarget_scores = compute_scores_samples(model, samples,labels)
    frr, far, thresholds= compute_det_curve(target_scores, nontarget_scores)
    return frr, far, thresholds

def plot_det_curve(model, loader):
    # Compute DET curve
    frr, far, thresholds = compute_det_curve_loader(model, loader)

    # Compute EER
    eer_index = np.nanargmin(np.abs(frr - far))
    eer = np.mean([frr[eer_index], far[eer_index]])

    # Plot DET curve
    plt.figure(figsize=(10, 10))
    plt.plot(far, frr, label='DET Curve')
    plt.scatter(far[eer_index], frr[eer_index], color='red')  # EER point
    plt.text(far[eer_index], frr[eer_index], f'EER = {eer:.3f}')

    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Detection Error Trade-off (DET) Curve')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()

import pandas as pd

###DCF###
# https://github.com/kaldi-asr/kaldi/blob/master/egs/sre08/v1/sid/compute_min_dcf.py
#used default values from https://www.researchgate.net/publication/222679587_Application-independent_evaluation_of_speaker_detection are c_miss = 10,p_target = 0.01 and c_fa = 1
def ComputeMinDcf(fnrs, fprs, thresholds, p_target = cost_model['Ptar'], c_miss = cost_model['Cmiss'], c_fa = cost_model['Cfa']):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold



 ###Tandem DCF###

def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit(
            'ERROR: Your prior probabilities should be positive and sum up to one.'
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            'ERROR: you should provide miss rate of spoof tests against your ASV system.'
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit(
            'ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.
              format(cost_model['Ptar']))
        print(
            '   Pnon         = {:8.5f} (Prior probability of nontarget user)'.
            format(cost_model['Pnon']))
        print(
            '   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.
            format(cost_model['Pspoof']))
        print(
            '   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'
            .format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'
            .format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'
            .format(cost_model['Cfa_cm']))
        print(
            '   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'
            .format(cost_model['Cmiss_cm']))
        print(
            '\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)'
        )

        if C2 == np.minimum(C1, C2):
            print(
                '   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(
                    C1 / C2))
        else:
            print(
                '   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(
                    C2 / C1))

    return tDCF_norm, CM_thresholds

 #based on https://github.com/clovaai/aasist/blob/main/evaluation.py#L78
def calculate_tDCF_EER(target_scores, nontarget_scores, detector_target_scores, detector_nontarget_scores, cost_model, printout=True):
    # Fix tandem detection cost function (t-DCF) parameters
    # Assuming cost_model is provided as an argument and includes necessary parameters

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = compute_eer(target_scores, nontarget_scores)
    eer_cm = compute_eer(detector_target_scores, detector_nontarget_scores)[0]

    # Compute error rates for ASV
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(target_scores, nontarget_scores, nontarget_scores, asv_threshold)  # Assuming spoof scores for ASV are equivalent to nontarget_scores for simplicity

    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(detector_target_scores, detector_nontarget_scores, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost=False)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        # Assuming printout logic remains similar, adjust as necessary
        print(f'CM SYSTEM EER: {eer_cm * 100} %')
        print(f'min-tDCF: {min_tDCF}')

    return eer_cm, min_tDCF

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv



def mintdcf_loader(model,detector,loader_model,loader_detector):
    combined_model = CombinedDetectorModel(detector,model)
    target_scores, nontarget_scores = compute_scores_loader(combined_model,loader_model)
    detector_target_scores, detector_nontarget_scores = compute_scores_loader(detector, loader_detector)
    eer,min_tdcf = calculate_tDCF_EER(target_scores, nontarget_scores, detector_target_scores, detector_nontarget_scores, cost_model, printout=True)

###COMPARE MODELS###

def compare_models(models, model_names, test_loader, criterion):
    # Initialize a dictionary to hold metrics for all models
    metrics = {'Metric': ['Accuracy','AUC', 'EER', 'FAR (at EER)', 'FRR (at EER)', 'minDCF']}

    for model in models:
        # Compute metrics for the current model
        loss, accuracy = torch_test_phase(model, test_loader, criterion)
        eer, eer_threshold = compute_eer_loader(model, test_loader)
        frrs, fars, thresholds = compute_det_curve_loader(model, test_loader)
        auc = compute_auc(model,test_loader)

        # Compute minDCF
        min_dcf, min_dcf_threshold = ComputeMinDcf(frrs, fars, thresholds)

        # Find the index of the threshold closest to the EER threshold
        threshold_index = np.argmin(np.abs(thresholds - eer_threshold))

        # Use the index to find the FAR and FRR at the EER threshold
        far_at_eer = fars[threshold_index]
        frr_at_eer = frrs[threshold_index]

        # Round metrics for readability
        accuracy = round(accuracy, 3)
        eer = round(eer, 3)
        far_at_eer = round(far_at_eer, 3)
        frr_at_eer = round(frr_at_eer, 3)
        min_dcf = round(min_dcf, 3)

        # Store metrics using the model's name from model_names dictionary
        metrics[model_names[model]] = [
            accuracy,
            auc,
            eer,
            far_at_eer,
            frr_at_eer,
            min_dcf
        ]

    # Create a DataFrame to hold the comparison
    results_df = pd.DataFrame(metrics)

    # Return the comparison DataFrame
    return results_df


#From Speaker Guard
#modified
def preprocess(benign_xx, bits=16):
    # Check if benign_xx is a PyTorch tensor
    if isinstance(benign_xx, torch.Tensor):
        benign_xx = benign_xx.detach().cpu().numpy()

    # Continue with the preprocessing
    if not LOWER <= benign_xx.max() <= UPPER:
        benign_xx = benign_xx / (2 ** (bits - 1))

    return benign_xx

def Lp(benign_xx, adver_xx, p, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    return LA.norm(adver_xx-benign_xx, p)

def L2(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 2, bits=bits)

def L0(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 0, bits=bits)

def L1(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, 1, bits=bits)

def Linf(benign_xx, adver_xx, bits=16):
    return Lp(benign_xx, adver_xx, np.infty, bits=bits)

def SNR(benign_xx, adver_xx, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    noise = adver_xx - benign_xx

    power_noise = np.sum(noise ** 2)

    #print("power noise",power_noise)

    if power_noise <= 0.:
        return None
        #return np.infty
    power_benign = np.sum(benign_xx ** 2)
    snr = 10 * np.log10(power_benign / power_noise)
    return snr

def PESQ(benign_xx, adver_xx, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    pesq_value = pesq(16_000, benign_xx, adver_xx, 'wb' if bits == 16 else 'nb')
    return pesq_value

def STOI(benign_xx, adver_xx, fs=16_000, bits=16):
    benign_xx = preprocess(benign_xx, bits=bits)
    adver_xx = preprocess(adver_xx, bits=bits)
    d = stoi(benign_xx, adver_xx, fs, extended=False)
    return d

def get_all_metric(benign_xx, adver_xx, fs=16_000, bits=16):
    return [L2(benign_xx, adver_xx, bits),
            L0(benign_xx, adver_xx, bits),
            L1(benign_xx, adver_xx, bits),
            Linf(benign_xx, adver_xx, bits),
            SNR(benign_xx, adver_xx, bits),
            PESQ(benign_xx, adver_xx, bits),
            STOI(benign_xx, adver_xx, fs, bits)]

