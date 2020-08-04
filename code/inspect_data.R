library(data.table)
library(ggplot2)

rm(list=ls())

d <- fread('../datta/Bayes_SML_EXP1.csv')
d[, PHASE := factor(PHASE, levels=c('Baseline', 'Adaptation', 'Washout'))]
d[, TRIAL := 1:.N, .(SUBJECT)]

dd <- d[, mean(HA_INITIAL), .(GROUP, PHASE, TRIAL, ROT)]
ggplot(dd, aes(TRIAL, V1, colour=factor(GROUP))) +
    geom_point(size=5) +
    geom_line(aes(TRIAL, -ROT), colour='black', size=2)

dd <- d[, mean(HA_END), .(GROUP, PHASE, TRIAL, ROT)]
ggplot(dd, aes(TRIAL, V1, colour=factor(GROUP))) +
    geom_point(size=5) +
    geom_line(aes(TRIAL, -ROT), colour='black')


ggplot(d, aes(TRIAL, HA_INITIAL, colour=factor(GROUP))) +
    geom_point(size=5) +
    geom_line(aes(TRIAL, -ROT), colour='black') +
    facet_wrap(~SUBJECT)


ggplot(d, aes(TRIAL, HA_END, colour=factor(GROUP))) +
    geom_point(size=5) +
    geom_line(aes(TRIAL, -ROT), colour='black') +
    facet_wrap(~SUBJECT)



dd <- d[, .(mean(HA_INITIAL), mean(HA_END)), .(GROUP, PHASE, TRIAL, ROT)]
ggplot(data=dd) +
    geom_point(data=dd, aes(TRIAL, V1, size=5), colour='red') +
    geom_point(data=dd, aes(TRIAL, V2, size=5), colour='blue') +
    geom_line(data=dd, aes(TRIAL, -ROT), colour='black', size=2) +
    facet_wrap(~GROUP)
