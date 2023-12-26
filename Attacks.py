

class Attacks:

    def __init__(self, classifier):
        self.classifier = classifier
        self.transfer_attack_dic = None


    def classifier_to_attack(self,model,clip_values,x,y):
        self.classifier = KerasClassifier(model=self.classifier,clip_values=clip_values, use_logits=False)
        return self.classifier


    def set_params(self,params):
        self.attacker.set_params(**params)
        self.attacker.set_params(**{'verbose': False})


    def attack(self, samples_to_attack,target_labels=None):
        if target_labels is None:
            samples_adv = self.attacker.generate(samples_to_attack)
        else:
            samples_adv = self.attacker.generate(samples_to_attack,target_labels)
        return samples_adv


    def top_attacks(self,model, classifier, combinations, samples_labels, samples_to_attack, N_top=10,scaler=None):

        if N_top>=len(combinations):
            N_top = len(combinations)

        params_accuracy = {}
        params_samples = {}

        if os.path.exists(self.dict_path):
            attacks_dict = load_dict_from_pickle(self.dict_path)

        for param in tqdm(combinations,desc="searching top "+str(N_top)+" attacks"):

            if os.path.exists(self.dict_path) and str(param) in attacks_dict.keys():

                samples_adv = attacks_dict[str(param)]
            else:
                self.attacker.set_params(**param)
                self.attacker.set_params(**{'verbose': False})

                if "targeted" in dict(param).keys() and dict(param)["targeted"]==1:
                    targeted_samples = tf.keras.utils.to_categorical(np.ones(len(samples_labels)),num_classes=2)
                    samples_adv = self.attack(samples_to_attack,targeted_samples)
                else:
                    samples_adv = self.attack(samples_to_attack)



                if os.path.exists(self.dict_path):
                    attacks_dict[str(param)] = samples_adv
                    save_dict_to_pickle(attacks_dict,self.dict_path)

            if scaler is not None:
                samples_adv = scaler.transform(samples_adv)

            params_accuracy[str(param)] = model.evaluate(samples_adv, samples_labels)
            params_samples[str(param)] = samples_adv


        top_combinations = []
        top_samples = []
        while len(top_combinations) < N_top and len(params_accuracy)>0:
            min_key = min(params_accuracy, key=params_accuracy.get)
            top_combinations.append(eval(min_key))
            top_samples.append(params_samples[min_key])
            params_accuracy.pop(min_key)

        top_combinations.reverse()
        top_samples.reverse()

        return top_combinations,top_samples

    def transfer_setup(self,models, type_of_attacks):
        self.transfer_attack_dic = {}
        for mod in models:
            self.transfer_attack_dic[mod] = {}
            for att in type_of_attacks:
                self.transfer_attack_dic[mod][att] = []

        return self.transfer_attack_dic


    def is_one_hot(self,array):
        # Check if the array is 2D
        if array.ndim != 2:
            return False
        # Check each row has one '1' and the rest '0's
        for row in array:
            if np.sum(row) != 1 or np.any((row != 0) & (row != 1)):
                return False
        return True


    def transfer_attack(self,samples_adv, samples_labels, models, types_of_defenses):

        if self.transfer_attack_dic is None:
            self.transfer_attack_dic = self.transfer_setup(models,types_of_defenses)

        for mod in models:
            for defense in types_of_defenses:

                #apply scaler
                if "S" in defense:
                    samples_adv = self.apply_scaler(samples_adv)
                    predictions = mod.predict(samples_adv)
                    predictions = np.round(predictions).astype(int)
                    if not self.is_one_hot(predictions):
                        predictions = tf.keras.utils.to_categorical(predictions,num_classes=2)
                    accuracy = accuracy_score(samples_labels,predictions)
                #normal behavior
                if "N" in defense:
                    predictions = mod.predict(samples_adv)
                    predictions = np.round(predictions).astype(int)
                    if not self.is_one_hot(predictions):
                        predictions = tf.keras.utils.to_categorical(predictions,num_classes=2)
                    accuracy = accuracy_score(samples_labels,predictions)
                #apply detector
                if "D" in defense:
                    predictions = self.apply_detector(samples_adv,mod)
                    predictions = np.round(predictions).astype(int)
                    accuracy = accuracy_score(samples_labels, predictions)
                #apply adversarial
                if "A" in defense:
                    predictions = self.apply_adversarial(samples_adv)
                    predictions = np.round(predictions).astype(int)
                    accuracy = accuracy_score(samples_labels, predictions)

                self.transfer_attack_dic[mod][defense].append(accuracy)
        return self.transfer_attack_dic


    def parameters(params):
        keys, values = zip(*params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts

    def plot_results(axis_x, labels, title, x_label, y_label, colors, model, models,models_name,transfer, transfer_attack,scaler=None):
        figure, axis = plt.subplots(1, len(models), figsize=(32,10))
        labels_reference = {'N': 'no defense', 'S': 'stand', 'D': 'detector', 'A': 'adversarial'}

        if scaler is not None:
            labels_reference = {'N': 'standardization', 'S': 'stand', 'D': 'standardization+detector', 'A': 'standardization+adversarial'}

        print(transfer_attack)
        for m in models:
            for defense, accuracies in transfer_attack[m].items():
                for other_defense in transfer_attack[m].keys():
                    if defense==other_defense:continue
                    equal_inds = [i for i, (x, y) in enumerate(zip(accuracies,transfer_attack[m][other_defense])) if x == y]
                    if len(equal_inds)>0:
                        for ind in equal_inds:
                            transfer_attack[m][other_defense][ind]=accuracies[ind]-0.005
        print(transfer_attack)


        for j in range(len(labels)):
            l = ' '
            for index in range(len(labels[j])):
                l = l + labels_reference[labels[j][index]]
                if index < len(labels[j])-1: l = l + '-'
            axis[0].plot(axis_x, transfer_attack[model][labels[j]], color=tuple(colors[j]), label=str(l), linewidth = 2)
        axis[0].set_title(title,fontsize='xx-large')
        axis[0].legend(fontsize='xx-large')
        axis[0].set_xlabel(x_label,fontsize = 'xx-large')
        axis[0].set_ylabel(y_label,fontsize = 'xx-large')
        axis[0].tick_params(axis='both', which='major', labelsize='xx-large')
        for tick in axis[0].get_xticklabels():
            tick.set_rotation(90)
        axis[0].set_ylim([-0.05, 1.05])

        if not transfer:
            return
        i = 0
        for mod in models:
            if mod == model:
                continue
            for j in range(len(labels)):
                if transfer_attack[mod][labels[j]][0] < 0:
                    continue
                l = ' '
                for index in range(len(labels[j])):
                    l = l + labels_reference[labels[j][index]]
                    if index < len(labels[j])-1: l = l + '-'
                axis[i + 1].plot(axis_x, transfer_attack[mod][labels[j]], color=tuple(colors[j]), label=str(l),linewidth = 2)
            axis[i + 1].set_title("transferability on " + models_name[mod],fontsize='xx-large')
            axis[i + 1].legend(fontsize='xx-large')
            axis[i + 1].set_xlabel(x_label,fontsize = 'xx-large')
            axis[i + 1].set_ylabel(y_label,fontsize = 'xx-large')
            axis[i + 1].tick_params(axis='both', which='major', labelsize='xx-large')
            for tick in axis[i + 1].get_xticklabels():
                tick.set_rotation(90)
            axis[i + 1].set_ylim([-0.05, 1.05])
            i = i + 1

        #plt.tight_layout()
        plt.show()


    def create_colors(type_of_defenses):
        colors = []
        hue_values = np.linspace(0, 1, len(type_of_defenses), endpoint=False)
        np.random.shuffle(hue_values)

        for hue in hue_values:
            saturation = np.random.uniform(0.6, 0.9)
            value = np.random.uniform(0.7, 0.9)
            color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(color)

        return colors


    def set_detector(self,detector):
        self.detector = detector


    def apply_detector(self,samples_adv,model):

        final_predictions = []
        # Loop over the standardized test data

        for sample in samples_adv:

            # Reshape the sample to (1, -1) because the predict method expects a batch
            sample = sample.reshape(1, -1)
            # Use the detector to make a prediction
            detector_prediction = self.detector.predict(sample)

            # Check if the detector predicts the sample as clean (0)
            if np.argmax(detector_prediction) == 0:
                # If the detector predicts the sample as clean, pass it to the main model
                model_prediction = model.predict(sample)
                final_predictions.append([1-np.argmax(model_prediction),np.argmax(model_prediction)])

            else:
                # If the detector predicts the sample as adversarial (1), set the final prediction as 0
                final_predictions.append([1, 0])

        return final_predictions


    def set_adversarial(self,adversarial):
        self.adversarial = adversarial


    def apply_adversarial(self,samples_adv):
        adv_predicts = self.adversarial.predict(samples_adv)
        adversarial_predictions = [[1-np.argmax(x),np.argmax(x)] for x in adv_predicts ]
        return adversarial_predictions

    def set_scaler(self,scaler):
        self.scaler = scaler


    def apply_scaler(self,samples_adv):
        #print("before scaler:",samples_adv.shape,samples_adv)
        samples_adv = self.scaler.transform(samples_adv)
        #print("after scaler:",samples_adv.shape,samples_adv)
        return samples_adv


    def generate_random_samples(self,attacker,combinations,n_samples,samples_to_attack):
        n_combinations = len(combinations)
        samples_per_comb = round(n_samples) // n_combinations
        samples = np.empty((0, samples_to_attack.shape[1]))

        if os.path.exists(self.dict_path):
            attacks_dict = load_dict_from_pickle(self.dict_path)

        for comb in tqdm(combinations,desc="generating random samples from "+str(attacker)):

            if os.path.exists(self.dict_path) and str(comb) in attacks_dict.keys():
                samples_adv = attacks_dict[str(comb)]
            else:
                print(str(comb),attacks_dict.keys())
                attacker.set_params(comb)
                samples_adv = attacker.attack(samples_to_attack)
                if os.path.exists(self.dict_path):
                    attacks_dict[str(comb)] = samples_adv
                    save_dict_to_pickle(attacks_dict,self.dict_path)

            # Randomly select samples_per_comb elements from samples_adv
            random_samples = random.sample(list(samples_adv), min(samples_per_comb, len(samples_adv)))
            random_samples = np.array(random_samples)

            samples = np.vstack((samples, random_samples))

        return samples

    def set_dict_path(self,path):
        self.dict_path = path








class FGSM(Attacks):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.attacker = FastGradientMethod(estimator=classifier)


class BIM(Attacks):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.attacker = BasicIterativeMethod(estimator=classifier)


class PGD(Attacks):
    def __init__(self, classifier,):
        super().__init__(classifier)
        self.attacker = ProjectedGradientDescent(estimator=classifier)


class DF(Attacks):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.attacker = DeepFool(classifier)


class CW(Attacks):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.attacker = CarliniL2Method(classifier)
