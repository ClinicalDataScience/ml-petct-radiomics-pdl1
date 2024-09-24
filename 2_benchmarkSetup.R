# # load packages
# PACKAGES = c(
#   "mlr3",
#   "mlr3viz",
#   "mlr3pipelines",
#   "mlr3tuning",
#   "mlr3hyperband",
#   "mlr3learners",
#   "ranger",
#   "glmnet",
#   "praznik",
#   "FSelectorRcpp",
#   "mlr3tuning",
#   "bbotk",
#   "mlr3misc",
#   "paradox",
#   "dplyr",
#   "data.table",
#   "batchtools",
#   "mlr3batchmark",
#   "data.table"
# )
#
# # load all required packages
# sapply(PACKAGES, require, character.only = TRUE)

options("mlr3.debug" = TRUE)


#################################
# Preparation
# !(This step must be tailored to the individual data situation.)
#################################

# in our case: list with 3 elements = datasets
# of course you could load each dataset separatly
# each data set needs to contains the target "pdl1" and the id "pseudonym"
load("data/datasetList.RData")

# name tasks
tasks = c("clin", "f01", "lut")

# unlist data frames
for(name in names(datasetList)){
  assign(name, datasetList[[name]])
}
rm(name)


#################################
# Benchmark Setup
#################################

# generate list of tasks
tskList = list()

# define each mlr3 task within a loop
for(t in 1:length(tasks)){
  tskList[[tasks[t]]] = TaskClassif$new(id = tasks[t],
    backend = get(tasks[t]) %>% select(-c("pseudonym")),
    target = "pdl1", positive = "1")
  # stratification
  tskList[[t]]$col_roles$stratum = "pdl1"
}

# plot the distribution of the target "pdl1"
autoplot(tskList[[1]])

# task ids
task_ids = sapply(tskList, function(task) task$id)
tasks = setNames(tasks, task_ids)

# chose/set measure/metric
# here: AUC
measure = msr("classif.auc")
measureTuning = measure

# chose a list of learners
# here: featureless as baseline method, ranger = random forest, cv_glmnet = elastic net and xgboost = boosting technique
learners = list(
  lrn("classif.featureless", predict_type = "prob"),
  lrn("classif.ranger", predict_type = "prob"),
  lrn("classif.cv_glmnet", predict_type = "prob"),
  lrn("classif.xgboost", predict_type = "prob")
)

# list of learners analyzed in benchmark
learnerIDs = list(
  "classif.featureless",
  "classif.xgboost", "classif.xgboost-tuned",
  "classif.ranger", "classif.ranger-tuned",
  "classif.cv_glmnet", "classif.cv_glmnet-tuned"
)

# setting up resampling methods
resamplingOuter = rsmp("cv", folds = 10L)
resamplingInner = rsmp("cv", folds = 5L)

# Settings for tuners (hyperband and random search)
ETA = 2
N_EVALS = 200L


#################################
# Benchmark Pipeline, Functions
#################################

# tuner: for xgboost and ranger hyperband, in case of glmnet no meaningful eta definable -> use random search instead
getTuner = function(learner_id) {
  if (learner_id == "classif.cv_glmnet") {
    return(tnr("random_search"))
  }
  else {
    return(tnr("hyperband", eta = ETA))
  }
}

# function to create graph learner
getGraph = function(learner_id, learner){
  return(
    as_learner(
      po("scale") %>>%
        po("imputemode") %>>%
        po("encode") %>>%
        po("imputehist") %>>%
        po("removeconstants") %>>%
        po("fixfactors") %>>%
        po("learner", learner)
    )
  )
}

# function for extracting tuning values according to learner id
# hyperparameters and tuning space selected in accordance to https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1301
getTuningParams = function(learner_id, task) {
  if (learner_id == "classif.ranger") {
    search_space = ps(
      # set budget hyperparameter for hyperband tuner
      classif.ranger.num.trees = p_int(lower = 20L, upper = 1500L, tags = "budget"),
      classif.ranger.mtry = p_int(lower = 3L, upper = ceiling((task$ncol)^(1/1.5))),
      classif.ranger.min.node.size = p_int(lower = 1L, upper = task$nrow)
    )
  }
  if (learner_id == "classif.cv_glmnet") {
    search_space = ps(
      classif.cv_glmnet.alpha = p_dbl(lower = 0, upper = 1)
    )
  }
  if (learner_id == "classif.xgboost") {
    search_space = ps(
      # set budget hyperparameter for hyperband tuner
      classif.xgboost.nrounds = p_int(lower = 20L, upper = 1500L, tags = "budget"),
      classif.xgboost.eta = p_dbl(lower = -4L, upper = 0, trafo = function(x) 10^x),
      classif.xgboost.max_depth = p_int(lower = 1L, upper = 20),
      classif.xgboost.colsample_bylevel = p_dbl(lower = 0.1, upper = 1),
      classif.xgboost.lambda = p_int(lower = -10, upper = 10, trafo = function(x) 2^x),
      classif.xgboost.alpha = p_int(lower = -10, upper = 10, trafo = function(x) 2^x),
      classif.xgboost.subsample = p_dbl(lower = 0.1, upper = 1)
    )
  }
  return(search_space)
}

# function to create learners with according settings determined by the learner name (learner_id) provided
getLearner = function(learner_id, task) {
  # fallback_learner = lrn(FALLBACK_LEARNER_ID)
  id = stringr::str_split(learner_id, "-", simplify = TRUE)[1, ]
  TUNING = ifelse("tuned" %in% id, "Tuned", "Untuned")
  learner_id = id[1]
  learner = lrn(learner_id, predict_type = "prob")

  # graphlearner
  glearner = getGraph(learner_id, learner)
  glearner$id = paste(learner_id, TUNING, sep = "_") #BALANCE

  # tuning
  if (TUNING == "Tuned") {
    search_space = getTuningParams(learner_id, task)

    # autotuner for hyperparameters, search space and tuner/terminator depending on learner
    at = AutoTuner$new( # autotuner object is named with an additional ".tuned" in the end
      learner = glearner,
      resampling = resamplingInner,
      measure = measureTuning,
      search_space = search_space,
      tuner = tnr("random_search"),
      terminator = trm("evals", n_evals = N_EVALS),
      store_tuning_instance = TRUE,
      store_benchmark_result = TRUE,
      store_models = TRUE
    )
    return (at)
  } else {
    # no tuning
    return(glearner)
  }
}


#################################
# Benchmark Design
#################################

# list of learners conatained in learnerIDs
learners = unlist(lapply(tskList, function(t) lapply(learnerIDs, function(l) {
  learner = getLearner(l, t)
  learner$id = paste(learner$id, t$id, sep = "_")
  return(learner)})))

# resampling plus instantiation
set.seed(2223)
resamplings = resamplingOuter$clone(deep = TRUE)$instantiate(tskList[[1]])

# design of the benchmark/batchmark, custom
design = data.table(
  task = rep(tskList, each = length(learnerIDs)),
  learner = learners,
  resampling = rep(list(resamplings), length(learners))
)

# here: benchmark() function, but for cluster use batchmark()
bmr = benchmark(design, store_models = TRUE)
# save(bmr, file = "results/bmr.RData")

# bmr$aggregate(measure)
# autoplot(bmr)
# autoplot(bmr, type = "roc")
