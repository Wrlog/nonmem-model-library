# Fit the simulated datasets with nlmixr2.
#
# The control streams in models/ are written for NONMEM, which is licensed
# and not run here. nlmixr2 is open source and estimates the same population
# models by FOCEi, so the library can show real estimation results -- how
# close the estimates land to the parameters the data was simulated from,
# goodness of fit, and a visual predictive check -- rather than only the
# simulation.
#
# Each model below is the same structure as the matching .mod file. Where
# the two must agree, they are commented so a reader can check.
#
# Writes, per model, into fit/results/:
#   <model>_estimates.csv  parameter, estimate, se, rse, shrinkage
#   <model>_gof.csv        ID TIME DV PRED IPRED CWRES IWRES
#   <model>_vpc.csv        simulated replicates for the VPC
#   <model>_status.json    convergence, objective function, run time

suppressPackageStartupMessages({
  library(nlmixr2)
  library(dplyr)
  library(jsonlite)
})

out_dir <- file.path("fit", "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# A "." in a numeric column is how NONMEM marks "not applicable"; nlmixr2
# wants NA or 0.
read_nm <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE, na.strings = c(".", "NA", ""))
  for (col in names(d)) {
    if (is.character(d[[col]])) {
      converted <- suppressWarnings(as.numeric(d[[col]]))
      if (!all(is.na(converted[!is.na(d[[col]])]))) d[[col]] <- converted
    }
  }
  if ("AMT" %in% names(d)) d$AMT[is.na(d$AMT)] <- 0
  if ("RATE" %in% names(d)) d$RATE[is.na(d$RATE)] <- 0
  if ("EVID" %in% names(d)) d$EVID[is.na(d$EVID)] <- 0
  d
}

save_fit <- function(fit, key, started, method = "focei") {
  est <- as.data.frame(fit$parFixedDf)
  est$parameter <- rownames(est)
  write.csv(est, file.path(out_dir, paste0(key, "_estimates.csv")),
            row.names = FALSE)

  gof <- as.data.frame(fit)
  keep <- intersect(c("ID", "TIME", "DV", "PRED", "IPRED", "CWRES", "IWRES",
                      "RES", "CPRED"), names(gof))
  write.csv(gof[, keep], file.path(out_dir, paste0(key, "_gof.csv")),
            row.names = FALSE)

  status <- list(
    model = key,
    objective = tryCatch(as.numeric(fit$objDf$OBJF[1]), error = function(e) NA),
    message = tryCatch(as.character(fit$message), error = function(e) ""),
    seconds = round(as.numeric(difftime(Sys.time(), started, units = "secs")), 1),
    method = method,
    engine = paste0("nlmixr2 ", as.character(utils::packageVersion("nlmixr2")))
  )
  write_json(status, file.path(out_dir, paste0(key, "_status.json")),
             auto_unbox = TRUE, pretty = TRUE)

  # VPC: simulate from the fit and write the replicates long. The percentiles
  # are computed in Python so the plotting stays in one place.
  vpc <- tryCatch({
    sim <- vpcSim(fit, n = 200, seed = 1234)
    as.data.frame(sim)[, intersect(c("sim.id", "id", "time", "sim"),
                                   names(as.data.frame(sim)))]
  }, error = function(e) {
    message("  vpcSim failed for ", key, ": ", conditionMessage(e))
    NULL
  })
  if (!is.null(vpc)) {
    write.csv(vpc, file.path(out_dir, paste0(key, "_vpc.csv")), row.names = FALSE)
  }
  invisible(status)
}

# --------------------------------------------------------------------------
# 1. Two-compartment IV  (models/pk_2cmt_iv.mod)
# --------------------------------------------------------------------------

pk_2cmt <- function() {
  ini({
    tvcl <- log(5.0)    # THETA(1)
    tvv1 <- log(15.0)   # THETA(2)
    tvq  <- log(3.0)    # THETA(3)
    tvv2 <- log(25.0)   # THETA(4)
    eta.cl ~ 0.09       # OMEGA(1,1)
    eta.v1 ~ 0.06       # OMEGA(2,2)
    prop.err <- 0.15    # SIGMA(1,1), as a CV
  })
  model({
    # Allometry on 70 kg, exponents fixed as in the control stream.
    cl <- exp(tvcl + eta.cl) * (WT / 70)^0.75
    v1 <- exp(tvv1 + eta.v1) * (WT / 70)
    q  <- exp(tvq) * (WT / 70)^0.75
    v2 <- exp(tvv2) * (WT / 70)
    linCmt() ~ prop(prop.err)
  })
}

# --------------------------------------------------------------------------
# 2. Claret tumour growth inhibition  (models/tgi_claret.mod)
# --------------------------------------------------------------------------

tgi_claret <- function() {
  ini({
    tvy0  <- log(50.0)
    tvkl  <- log(0.006)
    tvkd  <- log(0.0004)
    tvlam <- log(0.015)
    eta.y0 ~ 0.18
    eta.kl ~ 0.22
    eta.kd ~ 0.30
    prop.err <- 0.12
  })
  model({
    y0  <- exp(tvy0 + eta.y0)
    kl  <- exp(tvkl + eta.kl)
    kd  <- exp(tvkd + eta.kd)
    lam <- exp(tvlam)
    tumour(0) <- y0
    d/dt(tumour) <- kl * tumour - kd * EXPO * exp(-lam * t) * tumour
    tumour ~ prop(prop.err)
  })
}

# --------------------------------------------------------------------------
# 3. Indirect response, inhibition of production
#    (models/pkpd_idr_inhibition.mod)
# --------------------------------------------------------------------------

idr_inhibition <- function() {
  ini({
    tvcl   <- log(4.0)
    tvv    <- log(30.0)
    tvkin  <- log(10.0)
    tvkout <- log(0.10)
    tvic50 <- log(8.0)
    logit.imax <- 1.4      # logit(0.8), keeps IMAX in (0, 1)
    eta.kout ~ 0.09
    eta.ic50 ~ 0.22
    prop.err <- 0.10
  })
  model({
    cl   <- exp(tvcl)
    v    <- exp(tvv)
    kin  <- exp(tvkin)
    kout <- exp(tvkout + eta.kout)
    ic50 <- exp(tvic50 + eta.ic50)
    imax <- 1 / (1 + exp(-logit.imax))

    # Baseline is the untreated steady state, as in the control stream.
    response(0) <- kin / kout

    conc <- central / v
    inh  <- 1 - imax * conc / (ic50 + conc)
    d/dt(central)  <- -(cl / v) * central
    d/dt(response) <- kin * inh - kout * response

    response ~ prop(prop.err)
  })
}

# --------------------------------------------------------------------------

jobs <- list(
  list(key = "pk_2cmt_iv", model = pk_2cmt, est = "focei",
       data = function() {
         d <- read_nm(file.path("data", "pk_2cmt_iv.csv"))
         d$CMT <- NULL          # single output, linCmt handles the dosing
         d
       }),
  # SAEM for the tumour model: three random effects with 45-60% CV on nine
  # observations per subject is exactly the case where FOCEi's linearisation
  # struggles, and the first run of this library showed it -- the residual
  # error came back three times its simulated value.
  list(key = "tgi_claret", model = tgi_claret, est = "saem",
       data = function() {
         d <- read_nm(file.path("data", "tgi_claret.csv"))
         d$AMT <- 0; d$EVID <- 0
         d
       }),
  list(key = "pkpd_idr_inhibition", model = idr_inhibition, est = "focei",
       data = function() {
         d <- read_nm(file.path("data", "pkpd_idr_inhibition.csv"))
         # Dose records go to the central compartment, observations to the
         # response compartment.
         d$CMT <- ifelse(d$EVID == 1, "central", "response")
         d
       })
)

failures <- 0
for (job in jobs) {
  cat("\nfitting", job$key, "\n")
  started <- Sys.time()
  ok <- tryCatch({
    dat <- job$data()
    method <- if (is.null(job$est)) "focei" else job$est
    control <- if (method == "saem") {
      saemControl(print = 0, nBurn = 300, nEm = 400)
    } else {
      foceiControl(print = 0, maxOuterIterations = 200)
    }
    fit <- nlmixr2(job$model, dat, est = method, control = control)
    st <- save_fit(fit, job$key, started, method)
    cat("  OFV", st$objective, "in", st$seconds, "s\n")
    TRUE
  }, error = function(e) {
    message("  FAILED: ", conditionMessage(e))
    FALSE
  })
  if (!ok) failures <- failures + 1
}

cat("\n", length(jobs) - failures, "of", length(jobs), "models fitted\n")
if (failures > 0) quit(status = 1)
