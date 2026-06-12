using Pkg
Pkg.activate("power_grids")

using PowerDynamics
using OrderedCollections
using SyntheticPowerGrids

nodal_parameters = Dict(:η => 3e-3, :α => 5.0, :κ => π / 2)

## Buses for load flow
buses_loadflow = OrderedDict(
    "bus1" => PVAlgebraic(P = 2.32, V = 1),
    "bus2" => SlackAlgebraic(U = 1),
    "bus3" => PVAlgebraic(P = -0.942, V = 1),
    "bus4" => PQAlgebraic(P = -0.478, Q = -0.0),
    "bus5" => PQAlgebraic(P = -0.076, Q = -0.016),
    "bus6" => PVAlgebraic(P = -0.122, V = 1),
    "bus7" => PQAlgebraic(P = -0.0, Q = -0.0),
    "bus8" => PVAlgebraic(P = 0.0, V = 1),
    "bus9" => PQAlgebraic(P = -0.295, Q = -0.166),
    "bus10" => PQAlgebraic(P = -0.09, Q = -0.058),
    "bus11" => PQAlgebraic(P = -0.035, Q = -0.018),
    "bus12" => PQAlgebraic(P = -0.061, Q = -0.016),
    "bus13" => PQAlgebraic(P = -0.135, Q = -0.058),
    "bus14" => PQAlgebraic(P = -0.149, Q = -0.05),
);

#! format: off
# Transmission lines
branches=OrderedDict(
    "branch1" => StaticLine(from = "bus1", to = "bus2", Y=4.999131600798035-1im*15.263086523179553),
    "branch2" => StaticLine(from = "bus1", to = "bus5", Y=1.025897454970189-1im*4.234983682334831),
    "branch3" => StaticLine(from = "bus2", to = "bus3", Y=1.1350191923073958-1im*4.781863151757718),
    "branch4" => StaticLine(from = "bus2", to = "bus4", Y=1.686033150614943-1im*5.115838325872083),
    "branch5" => StaticLine(from = "bus2", to = "bus5", Y=1.7011396670944048-1im*5.193927397969713),
    "branch6" => StaticLine(from = "bus3", to = "bus4", Y=1.9859757099255606-1im*5.0688169775939205),
    "branch7" => StaticLine(from = "bus4", to = "bus5", Y=6.840980661495672-1im*21.578553981691588),
    "branch8" => StaticLine(from = "bus4", to = "bus7", Y=0.0-1im*4.781943381790359),
    "branch9" => StaticLine(from = "bus4", to = "bus9", Y=0.0-1im*1.7979790715236075),
    "branch10" => StaticLine(from = "bus5", to = "bus6", Y=0.0-1im*3.967939052456154),
    "branch11" => StaticLine(from = "bus6", to = "bus11", Y=1.9550285631772604-1im*4.0940743442404415),
    "branch12" => StaticLine(from = "bus6", to = "bus12", Y=1.525967440450974-1im*3.1759639650294003),
    "branch13" => StaticLine(from = "bus6", to = "bus13", Y=3.0989274038379877-1im*6.102755448193116),
    "branch14" => StaticLine(from = "bus7", to = "bus8", Y=0.0-1im*5.676979846721544),
    "branch15" => StaticLine(from = "bus7", to = "bus9", Y=0.0-1im*9.09008271975275),
    "branch16" => StaticLine(from = "bus9", to = "bus10", Y=3.902049552447428-1im*10.365394127060915),
    "branch17" => StaticLine(from = "bus9", to = "bus14", Y=1.4240054870199312-1im*3.0290504569306034),
    "branch18" => StaticLine(from = "bus10", to = "bus11", Y=1.8808847537003996-1im*4.402943749460521),
    "branch19" => StaticLine(from = "bus12", to = "bus13", Y=2.4890245868219187-1im*2.251974626172212),
    "branch20" => StaticLine(from = "bus13", to = "bus14", Y=1.1369941578063267-1im*2.314963475105352),
);
#! format: on

## Find load Flow
pg_loadflow = PowerGrid(buses_loadflow, branches)
op_loadflow = find_operationpoint(pg_loadflow)

Q_vec = op_loadflow[:, :q]

## Full buses with load flow from prevoius set-up
buses = OrderedDict(
    "bus1" => get_dVOCapprox(2.32, Q_vec[1], 1.0, nodal_parameters),
    "bus2" => SlackAlgebraic(U = 1),
    "bus3" => get_dVOCapprox(-0.942, Q_vec[3], 1.0, nodal_parameters),
    "bus4" => PQAlgebraic(P = -0.478, Q = -0.0),
    "bus5" => PQAlgebraic(P = -0.076, Q = -0.016),
    "bus6" => get_dVOCapprox(-0.122, Q_vec[6], 1.0, nodal_parameters),
    "bus7" => PQAlgebraic(P = -0.0, Q = -0.0),
    "bus8" => get_dVOCapprox(0.0, 0.0, Q_vec[8], nodal_parameters),
    "bus9" => PQAlgebraic(P = -0.295, Q = -0.166),
    "bus10" => PQAlgebraic(P = -0.09, Q = -0.058),
    "bus11" => PQAlgebraic(P = -0.035, Q = -0.018),
    "bus12" => PQAlgebraic(P = -0.061, Q = -0.016),
    "bus13" => PQAlgebraic(P = -0.135, Q = -0.058),
    "bus14" => PQAlgebraic(P = -0.149, Q = -0.05),
);

## Save grid to json
pg = PowerGrid(buses, branches)
op = find_operationpoint(pg) # Sanity-check: is there a fixed point?

write_powergrid(pg, joinpath(@__DIR__, "../power_grids/ieee14bus/ieee14bus.json"), Json)
