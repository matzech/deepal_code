from config import (
    n_init, n_query, 
    res_folder, 
    methods, methods_all,
    seeds,
    groups,
    #datasets
)


#############################################simulation settings###################################################################

#### Preparing pairs
k = [k for k,v in groups.items()]
import itertools
all_perms = list(itertools.permutations(k, 2))
base_model_trained_ons = [i[0] for i in all_perms]
active_learning_ds = [i[1] for i in all_perms]

print(all_perms)
print(base_model_trained_ons)


datasets=["controlled","all"]
n_rounds_dict = {
    "all": 150,
    "controlled": 75
}

n_rounds_from_scratch_dict = {
    "all": 215, # so it fits in time limit of 24h
    "controlled": 150
}


#######################################################cluster settings#################################################################
#### Memory
mem_mbs = {
    "RandomSampling": 100*1000,
    "BALDDropout": 100*1000, 
    "EntropySampling": 100*1000,
    "KCenterGreedy": 100*1000
}

### GPU settings
gpus_per_task = {
    "RandomSampling": "'volta|ampere'", #"'volta|pascal'",
    "EntropySampling": "'volta|ampere'",
    "BALDDropout": "'ampere'",
    "KCenterGreedy": "'ampere'",
}

rule run_all_simulations:
# double parentheses indicate the belonging to the parent expand
    input:
        #### subset of data
        expand(res_folder + "/{dataset}/from_scratch/{n_init}_{n_query}_{seed}/{method}/result.npz", dataset=datasets, method=methods_all, n_init=n_init, n_query=n_query, seed=seeds),
        expand(
            expand(res_folder + "/{{dataset}}/jointlearn/{base_model_trained_on}/{active_learning_ds}/{{n_query}}_{{seed}}/{{method}}/result.npz", zip,  base_model_trained_on=base_model_trained_ons, active_learning_ds=active_learning_ds), dataset=datasets,
            method=methods_all, n_query=n_query, seed=seeds),
        expand(
            expand(res_folder + "/{{dataset}}/finetune/{base_model_trained_on}/{active_learning_ds}/{{n_query}}_{{seed}}/{{method}}/result.npz", zip, base_model_trained_on=base_model_trained_ons, active_learning_ds=active_learning_ds), dataset=datasets,
            method=methods_all, n_query=n_query, seed=seeds),
        expand(res_folder + "/controlled/from_scratch/{n_init}_{n_query}_{seed}/{method}/result.npz", method=methods, n_init=n_init, n_query=n_query, seed=seeds),
        #### perfect models
        expand(res_folder + "/{dataset}/from_scratch/{n_init}_{n_query}_{seed}/{method}/results_perfect_3.npz", dataset=datasets, method=["RandomSampling"], n_init=n_init, n_query=n_query, seed=seeds),
        expand(
            expand(res_folder + "/{{dataset}}/jointlearn/{base_model_trained_on}/{active_learning_ds}/{{n_query}}_{{seed}}/{{method}}/results_perfect_2.npz", zip,  base_model_trained_on=base_model_trained_ons, active_learning_ds=active_learning_ds), dataset=datasets,
            method=["RandomSampling"], n_query=n_query, seed=seeds),
        expand(
            expand(res_folder + "/{{dataset}}/finetune/{base_model_trained_on}/{active_learning_ds}/{{n_query}}_{{seed}}/{{method}}/results_perfect_2.npz", zip, base_model_trained_on=base_model_trained_ons, active_learning_ds=active_learning_ds), dataset=datasets,
            method=["RandomSampling"], n_query=n_query, seed=seeds),
        ### make base models (single datasets already included)




################################################################################################################################################

######################## main rules

rule make_from_scratch:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_from_scratch_dict[w.dataset])
    output:
        res_folder + "/{dataset}/from_scratch/{n_init}_{n_query}_{seed}/{method}/result.npz"
    shell:
        """
        python main_from_scratch.py --dataset {wildcards.dataset} --n_round {params.n_rounds} --strategy_name {wildcards.method} --n_init_labeled 100 --n_query {wildcards.n_query} --res_file {output} --seed {wildcards.seed}
        """



rule make_fine_tuning:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_dict[w.dataset])
    input:
        base_model=res_folder + "/{dataset}/base_models/{base_model_trained_on}_model.pth"
    output:
        res_folder + "/{dataset}/finetune/{base_model_trained_on}/{active_learning_ds}/{n_query}_{seed}/{method}/result.npz"
    shell:
        """
        python main_finetune_joint_learn.py --dataset {wildcards.dataset} --n_round {params.n_rounds} --base_model_name {input.base_model} --base_model_trained_on {wildcards.base_model_trained_on} --active_learning_on {wildcards.active_learning_ds} \
            --strategy_name {wildcards.method} --n_query {wildcards.n_query} --res_file {output} --seed {wildcards.seed} --mode finetune --n_init_labeled 8 
        """

rule make_joint_learning:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_dict[w.dataset])
    input:
        base_model=res_folder + "/{dataset}/base_models/{base_model_trained_on}_model.pth"
    output:
        res_folder + "/{dataset}/jointlearn/{base_model_trained_on}/{active_learning_ds}/{n_query}_{seed}/{method}/result.npz"
    shell:
        """
        python main_finetune_joint_learn.py --dataset {wildcards.dataset} --n_round {params.n_rounds} --base_model_name {input.base_model} --base_model_trained_on {wildcards.base_model_trained_on} --active_learning_on {wildcards.active_learning_ds} \
            --strategy_name {wildcards.method} --n_query {wildcards.n_query} --res_file {output} --seed {wildcards.seed} --mode jointlearning --n_init_labeled 8
        """



####################################################################################################################################################################################
###################################################### perfect models
####################################################################################################################################################################################

rule make_from_scratch_perfect:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_dict[w.dataset])
    output:
        res_folder + "/{dataset}/from_scratch/{n_init}_{n_query}_{seed}/{method}/results_perfect_3.npz"
    shell:
        """
        python main_best_model_from_scratch.py --dataset {wildcards.dataset} --strategy_name {wildcards.method} --res_file {output} --seed {wildcards.seed}
        """


rule make_fine_tuning_perfect:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_dict[w.dataset])
    input:
        base_model=res_folder + "/{dataset}/base_models/{base_model_trained_on}_model.pth"
    output:
        res_folder + "/{dataset}/finetune/{base_model_trained_on}/{active_learning_ds}/{n_query}_{seed}/{method}/results_perfect_2.npz"
    shell:
        """
        python main_best_model_finetune_joint_learn.py --dataset {wildcards.dataset} --base_model_name {input.base_model} --base_model_trained_on {wildcards.base_model_trained_on} --active_learning_on {wildcards.active_learning_ds} \
           --res_file {output} --seed {wildcards.seed} --mode finetune 
        """


rule make_joint_learning_perfect:
    resources: 
        time_min=60*24*1, 
        mem_mb=(lambda w: mem_mbs[w.method]),
        cpus=12,
        constraint=(lambda w: gpus_per_task[w.method]),
    params:
        n_rounds=(lambda w: n_rounds_dict[w.dataset])
    input:
        base_model=res_folder + "/{dataset}/base_models/{base_model_trained_on}_model.pth"
    output:
        res_folder + "/{dataset}/jointlearn/{base_model_trained_on}/{active_learning_ds}/{n_query}_{seed}/{method}/results_perfect_2.npz"
    shell:
        """
        python main_best_model_finetune_joint_learn.py --dataset {wildcards.dataset}  --base_model_name {input.base_model} --base_model_trained_on {wildcards.base_model_trained_on} --active_learning_on {wildcards.active_learning_ds} \
           --res_file {output} --seed {wildcards.seed} --mode jointlearning 
        """
####################################################################################################################################################################################
###################################################### prepare base models as prerequisite for fine_tune, joint_learning
####################################################################################################################################################################################

rule prepare_base_model:
    resources: 
        time_min=60*24*1, 
        mem_mb=100000,
        cpus=12,
    output:
        base_model=res_folder + "/{dataset}/base_models/{base_model_trained_on}_model.pth"
    shell:
        """
        python main_train_base_model.py --dataset {wildcards.dataset} --res_file {output.base_model} --group {wildcards.base_model_trained_on}
        """
