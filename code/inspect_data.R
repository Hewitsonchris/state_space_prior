library(data.table)
library(ggplot2)

rm(list=ls())

d <- fread('../datta/Bayes_SML_EXP1_200820.csv')
d[, PHASE := factor(PHASE, levels=c('Baseline', 'Adaptation', 'Washout'))]
d[, TRIAL := 1:.N, .(SUBJECT)]

## dd <- d[, mean(HA_INIT), .(GROUP, PHASE, TRIAL, ROT)]
## ggplot(dd, aes(TRIAL, V1, colour=factor(GROUP))) +
##     geom_point(size=5) +
##     geom_line(aes(TRIAL, -ROT), colour='black', size=2)

## dd <- d[, mean(HA_END), .(GROUP, PHASE, TRIAL, ROT)]
## ggplot(dd, aes(TRIAL, V1, colour=factor(GROUP))) +
##     geom_point(size=5) +
##     geom_line(aes(TRIAL, -ROT), colour='black')

## ggplot(d, aes(TRIAL, HA_INIT, colour=factor(GROUP))) +
##     geom_point(size=5) +
##     geom_line(aes(TRIAL, -ROT), colour='black') +
##     facet_wrap(~SUBJECT)

## ggplot(d, aes(TRIAL, HA_END, colour=factor(GROUP))) +
##     geom_point(size=5) +
##     geom_line(aes(TRIAL, -ROT), colour='black') +
##     facet_wrap(~SUBJECT)

dd <- d[, .(mean(HA_INIT), mean(HA_END)), .(GROUP, PHASE, TRIAL, ROT)]
ggplot(data=dd) +
    geom_line(data=dd, aes(TRIAL, -ROT), colour='grey', size=0.5) +
    geom_point(data=dd, aes(TRIAL, V1), colour='red', size=1) +
    geom_point(data=dd, aes(TRIAL, V2), colour='blue', size=1) +
    facet_wrap(~GROUP) +
    theme(text = element_text(size = 12))
ggsave('../figures/means.pdf', width=10, height=5)

dd <- d[, .(mean(HA_INIT), mean(HA_END)), .(GROUP, SUBJECT, PHASE, TRIAL, ROT)]
ggplot(data=dd) +
    geom_line(data=dd, aes(TRIAL, -ROT), colour='grey', size=0.1) +
    geom_point(data=dd, aes(TRIAL, V1), colour='red', size=0.1) +
    geom_point(data=dd, aes(TRIAL, V2), colour='blue', size=0.1) +
    facet_wrap(~GROUP*SUBJECT, ncol=10) +
    theme(text = element_text(size = 6))
ggsave('../figures/subs.pdf', width=10, height=6)

dd <- d[TRIAL %in% 11:190, .(mean(HA_INIT), mean(HA_END)), .(GROUP, PHASE, TRIAL, ROT, SIG_MP)]
ggplot(data=dd) +
    geom_point(data=dd, aes(-ROT, V2, colour=factor(SIG_MP)), size=1) +
    geom_smooth(data=dd, aes(-ROT, V2, colour=factor(SIG_MP)), size=1, method='lm') +
    ylab('Endpoint HA') +
    facet_wrap(~GROUP) +
    theme(text = element_text(size = 6))

dd <- d[TRIAL %in% 11:190, .(mean(HA_INIT), mean(HA_END)), .(GROUP, PHASE, TRIAL, ROT, SIG_MP)]
dd <- dd[order(GROUP, TRIAL)]
ggplot(data=dd[SIG_MP != 0]) +
    geom_point(data=dd, aes(-shift(ROT), V1, colour=factor(SIG_MP)), size=5) +
    geom_smooth(data=dd, aes(-shift(ROT), V1, colour=factor(SIG_MP)), size=5, method='lm') +
    ylab('Initial HA') +
    facet_wrap(~GROUP) +
    theme(text = element_text(size = 50))
