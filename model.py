#######################################################################################
## Dependencies
#######################################################################################

# namedtuple creates a tuple with a name
# https://docs.python.org/3/library/collections.html#collections.namedtuple
from collections import namedtuple

# Model: https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
# Model is a class to embed modeling objects
from docplex.mp.model import Model
# LinearExpr contains the negate() function to create negate a linear expression
from docplex.mp.linear import LinearExpr

# used for normalisation of vectors
import numpy as np

# get all possible combinations for staticLex
import itertools

# Dependencies from diet.py to parse inputs and outputs
from docplex.util.environment import get_environment
from os.path import splitext
import os
import pandas
from six import iteritems
import sys

#######################################################################################
## My model
#######################################################################################
def optimize_offensive_army(axe_strength = 1, lc_strength = 1, ma_strength = 1, build_time_limit_seconds = 2_592_000, log_output = False): 
    # Initialise data

    Unit = namedtuple('Unit', ['name', 'att_strength', 'recruit_time_in_seconds', 'food'])

    axe_unit = Unit('axe', 45, 90, 1)
    lc_unit = Unit('lc', 130, 360, 4)
    ma_unit = Unit('ma', 150, 450, 5)
    serk_unit = Unit('serk', 300, 1200, 6)
    ram_unit = Unit('ram', 2, 480, 5)

    units = [axe_unit, lc_unit, ma_unit, serk_unit, ram_unit]
    barracks_units = [axe_unit, lc_unit, ma_unit, ram_unit]
    hoo_units = [serk_unit]

    with Model(name='Max strength unit time') as m:
        # Decision variable
        number_of_units = m.integer_var_dict(
            [u.name for u in units], name = 'number_of_units'
        )
        number_of_units

        # Decision expressions
        m.barracks_build_time = m.sum([number_of_units[u.name] * u.recruit_time_in_seconds for u in barracks_units])
        m.hoo_build_time = m.sum([number_of_units[u.name] * u.recruit_time_in_seconds for u in hoo_units])

        m.totalNegativeAttack = LinearExpr.negate(m.sum([number_of_units[u.name] * u.att_strength for u in units]))
        m.totalFood = m.sum([number_of_units[u.name] * u.food for u in units])

        # Constraints
        ct_food_no_church = m.add_constraints([m.totalFood <= 20596, m.totalFood >= 20580])
        ct_must_have_rams = m.add_constraint(number_of_units['ram'] >= 250)

        ct_build_time_less_than_4_weeks_barracks = m.add_constraints([1 <= m.barracks_build_time, (m.barracks_build_time <= build_time_limit_seconds)])
        ct_build_time_less_than_4_weeks_hoo = m.add_constraints([1 <= m.hoo_build_time, (m.hoo_build_time <= build_time_limit_seconds)])

        attack_strength_weights = [axe_strength, lc_strength, ma_strength]
        strength_ratios = attack_strength_weights / np.sum(attack_strength_weights)

        # Provide a range for the divided strength constraints
        lower_bound_cap = 0.95

        ct_axe_number = m.add_constraints([
            number_of_units['axe'] * axe_unit.att_strength <= -(m.totalNegativeAttack * strength_ratios[0]),
            number_of_units['axe'] * axe_unit.att_strength >= -(m.totalNegativeAttack * strength_ratios[0]) * lower_bound_cap
        ])

        ct_axe_number = m.add_constraints([
            number_of_units['lc'] * lc_unit.att_strength <= -(m.totalNegativeAttack * strength_ratios[1]),
            number_of_units['lc'] * lc_unit.att_strength >= -(m.totalNegativeAttack * strength_ratios[1]) * lower_bound_cap
        ])

        ct_axe_number = m.add_constraints([
            number_of_units['ma'] * ma_unit.att_strength <= -(m.totalNegativeAttack * strength_ratios[2]),
            number_of_units['ma'] * ma_unit.att_strength >= -(m.totalNegativeAttack * strength_ratios[2]) * lower_bound_cap
        ])

        # Objectives
        m.add_kpi(m.totalNegativeAttack, "Total negative attack strength")
        m.add_kpi(m.barracks_build_time, "Barracks build time")
        m.add_kpi(m.hoo_build_time, "Hall of Order build time")

        # Initialise KPI permutations
        kpis_permutations = itertools.permutations([m.totalNegativeAttack, m.hoo_build_time, m.barracks_build_time], 3)
        kpis_permutations = list(kpis_permutations)

        if log_output is True:
            # pretty names for results
            kpis_permutations_names = []
            for permutation in kpis_permutations:
                name = []
                for kpi in permutation:
                    if kpi is m.totalNegativeAttack:
                        name.append('totalNegativeAttack')
                    if kpi is m.hoo_build_time:
                        name.append('hoo_build_time')
                    if kpi is m.barracks_build_time:
                        name.append('barracks_build_time')
                kpis_permutations_names.append(name)

        # Solve
        solutions = []
        for idx, kpis in enumerate(kpis_permutations):
            m.minimize_static_lex(kpis)
            m.solve()

            solutions.append({
                "axe": number_of_units['axe'].solution_value,
                "lc": number_of_units['lc'].solution_value,
                "ma": number_of_units['ma'].solution_value,
                "serk": number_of_units['serk'].solution_value,
                "ram": number_of_units['ram'].solution_value,
                "food": sum([number_of_units[u.name].solution_value * u.food for u in units]),
                "time_in_seconds": sum([number_of_units[u.name].solution_value * u.recruit_time_in_seconds for u in units]),
                "time_in_days": round(sum([number_of_units[u.name].solution_value * u.recruit_time_in_seconds for u in units]) / (24 * 3600), 2),
                "total_attack_strength": -m.totalNegativeAttack.solution_value
            })

            if log_output is True:
                print(f"{kpis_permutations_names[idx]} - strength: {-m.totalNegativeAttack.solution_value}, food: {m.totalFood.solution_value}")
                print(
                    f"axe: {number_of_units['axe'].solution_value}"
                    f", lc: {number_of_units['lc'].solution_value}"
                    f", ma: {number_of_units['ma'].solution_value}"
                    f", serk: {number_of_units['serk'].solution_value}"
                    f", ram: {number_of_units['ram'].solution_value}"
                    f", food: {sum([number_of_units[u.name].solution_value * u.food for u in units])}"
                    f", time: {round(sum([number_of_units[u.name].solution_value * u.recruit_time_in_seconds for u in units]) / (24 * 3600), 2)} days"
                )
                print(' ')
        
        return solutions

#######################################################################################
## IBM Cloud's example code from diet.py
#######################################################################################
def get_all_inputs():
    '''Utility method to read a list of files and return a tuple with all
    read data frames.
    Returns:
        a map { datasetname: data frame }
    '''
    result = {}
    env = get_environment()
    print(env)
    for iname in [f for f in os.listdir('.') if splitext(f)[1] == '.csv']:
        with env.get_input_stream(iname) as in_stream:
            df = pandas.read_csv(in_stream)
            datasetname, _ = splitext(iname)
            result[datasetname] = df
    return result

# From reading the code here, it looks like each dictionary inside the `outputs` param
#   gets converted into a CSV
def write_all_outputs(outputs):
    '''Write all dataframes in ``outputs`` as .csv.

    Args:
        outputs: The map of outputs 'outputname' -> 'output df'
    '''
    for (name, df) in iteritems(outputs):
        csv_file = '%s.csv' % name
        print(csv_file)
        with get_environment().get_output_stream(csv_file) as fp:
            if sys.version_info[0] < 3:
                fp.write(df.to_csv(index=False, encoding='utf8'))
            else:
                fp.write(df.to_csv(index=False).encode(encoding='utf8'))
    if len(outputs) == 0:
        print("Warning: no outputs written")

#######################################################################################
## Prepare inputs
#######################################################################################
# Load CSV files into inputs dictionnary
# NOTE: Delete any empty CSVs from the directory if you're running this locally. 
#   Python will throw an error otherwise
inputs = get_all_inputs()

# Expect a params.csv
optimize_params = inputs['params'] # params is a table or a csv called params
axe_strength = optimize_params['axe_strength'].values[0]
lc_strength = optimize_params['lc_strength'].values[0]
ma_strength = optimize_params['ma_strength'].values[0]
build_time_limit_seconds = optimize_params['build_time_limit_seconds'].values[0]

#######################################################################################
## Solve
#######################################################################################
solutions = optimize_offensive_army(axe_strength, lc_strength, ma_strength, build_time_limit_seconds)

#######################################################################################
## Prepare outputs
#######################################################################################
solutions_df = pandas.DataFrame(solutions)
outputs = {}
outputs['solution'] = solutions_df

# Generate output files
write_all_outputs(outputs)
