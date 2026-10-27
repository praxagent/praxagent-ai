#!/usr/bin/env python3
"""Verify the main post's memory extension against pinned, copied receipts.

Run: python verify_memory_update.py --verify --research-repo /path/to/neural-shrimp
This checks the added prose and table values, copy hashes, plot data and exact
Git objects. It does not rerun the model or turn a numerical audit into a proof.
Historical post entries remain governed by the existing post-wide manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def field(obj, path):
    for part in path.split('.'):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def normal(text):
    return text.replace('**', '').replace('*', '').replace('−', '-').replace('–', '-')


def primitive_token_period(tokens):
    for period in range(1, len(tokens) + 1):
        if len(tokens) % period == 0 and all(tokens[i] == tokens[i % period] for i in range(len(tokens))):
            return period
    raise ValueError('token period unavailable')


def rendered(value, entry):
    op = entry.get('operation')
    if op == 'len':
        value = len(value)
    elif op == 'count-distinct-slices':
        value = len({row['transverse'] for row in value})
    elif op == 'max-root-tau':
        value = max(abs(row['tau']) for row in value)
    elif op == 'sum-section-evaluations':
        value = sum(row['evaluation_count'] for row in value)
    elif op == 'count-zero-offset-minima':
        value = sum(len(row['minima']) == 1 and row['minima'][0]['s'] == 0 for row in value)
    elif op == 'contrast-threshold-ratio':
        value = value['minimum_passed_finite_contrast'] / value['finite_contrast_threshold']
    elif op == 'max-abs-final-primary-trace':
        value = max(abs(c['decisive_iteration']['coarse_corrector']['equations'][0]) for c in value)
    elif op == 'max-abs-final-cyclic-trace':
        value = max(abs(p['tau']) for c in value for p in c['decisive_iteration']['cyclic_measurements']['phases'])
    elif op == 'max-abs-final-fplus':
        value = max(abs(c['decisive_iteration']['coarse_corrector']['equations'][1]) for c in value)
    elif op == 'feature-envelope-ratio':
        value = abs(value['value'])/value['empirical_total_error']
    elif op == '100-minus':
        value = 100 - value
    elif op == 'times-100':
        value = 100 * value
    elif op == 'primitive-token-period':
        value = primitive_token_period(value)
    elif op == 'token-repetitions':
        value = len(value) // primitive_token_period(value)
    elif op is not None:
        raise ValueError('unknown operation ' + op)
    style = entry['style']
    if style == 'raw':
        return str(value)
    if style == 'integer':
        require(value == int(value), 'noninteger count')
        return str(int(value))
    if style == 'word':
        return {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 10: 'ten', 12: 'twelve', 40: 'forty'}[value]
    if style == 'floor-word':
        return {5: 'five'}[math.floor(value)]
    if style == 'ceil-power10':
        require(value > 0, 'positive bound input required')
        exponent = math.ceil(math.log10(value))
        require(value < 10.0**exponent, 'strict power-of-ten bound fails')
        return '10^{' + str(exponent) + '}'
    if style == 'format':
        return format(value, entry['format'])
    if style == 'ceil-fixed':
        places = int(entry['format'])
        value = math.ceil(value * 10**places) / 10**places
        return format(value, f'.{places}f')
    if style == 'power10':
        exponent = round(math.log10(value))
        require(value == 10.0**exponent, 'not a power of ten')
        return '10^{' + str(exponent) + '}'
    if style == 'ceil-scientific':
        places = int(entry['format'])
        exponent = math.floor(math.log10(abs(value)))
        mantissa = math.ceil(value / 10**exponent * 10**places) / 10**places
        return format(mantissa, f'.{places}f') + r'\times10^{' + str(exponent) + '}'
    if style == 'scientific':
        mantissa, exponent = format(value, entry['format']).split('e')
        return mantissa + r'\times10^{' + str(int(exponent)) + '}'
    raise ValueError('unknown style ' + style)


def git_show(repo, commit, path):
    require(re.fullmatch('[0-9a-f]{40}', commit), 'exact commit required')
    require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'unsafe Git path')
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=repo)



def verify_direct(bundle, manifest, prose, repo):
    extension = next(e for e in manifest['extensions'] if e['id'] == 'memory-direct-20261022')
    receipts = {}
    for name, meta in extension['source_receipts'].items():
        data = (bundle / name).read_bytes()
        require(digest(data) == meta['sha256'] == manifest['receipts'][name], 'direct receipt hash drift')
        if repo:
            require(data == git_show(repo, meta['commit'], meta['source_path']), 'direct receipt differs from Git')
        receipts[name] = json.loads(data)
    for entry in extension['new_numbers']:
        require(entry in manifest['numbers'], 'direct entry absent from main manifest')
        require(entry['receipt_sha256'] == extension['source_receipts'][entry['receipt']]['sha256'], 'direct number hash drift')
        expected = rendered(field(receipts[entry['receipt']], entry['field']), entry)
        require(expected in normal(entry['appears_as']), 'direct display not rederived: ' + entry['id'])
        require(normal(entry['appears_as']) in prose, 'direct number absent from post: ' + entry['id'])
    receipt = receipts['memory-connection-probe-receipt.json']
    require(receipt['complete'] and receipt['counts']['unqualified_or_incomplete_points'] == 0, 'direct audit incomplete')
    measurements = receipt['measurements']
    require(measurements['all_measurement_stages_complete'] and not measurements['untested_stages'], 'direct stages incomplete')
    axes = measurements['before_after_axes']
    for case in axes:
        counts = []
        for axis in case['axes']:
            samples = sorted(axis['samples'], key=lambda row: row['offset'])
            counts.append(sum(a['point']['tau'] * b['point']['tau'] < 0 for a,b in zip(samples, samples[1:])))
        require(counts == [case['negative_axis_brackets'], case['positive_axis_brackets']], 'axis bracket count differs')
    require([(a['negative_axis_brackets'], a['positive_axis_brackets']) for a in axes] == [(2,0),(0,2)], 'opening-direction comparison differs')
    roots = measurements['trace_roots']
    require(len(roots) == receipt['counts']['trace_roots_found'] == receipt['counts']['roots_with_complete_offsets'], 'root count differs')
    for root in roots:
        require(root['status'] == 'complete' and root['cyclic_point']['phases'] == [0,6,12], 'root cyclic gate incomplete')
        require(root['tau'] == root['cyclic_point']['tau'], 'phase-zero root trace differs')
        require(len(root['radius_offsets']) == 2 and {row['offset'] for row in root['radius_offsets']} == {-0.003,0.003}, 'root offsets differ')
        for row in root['radius_offsets']:
            delta = row['point']['radius'] - root['radius']
            require(math.isclose(delta, row['radius_difference'], rel_tol=1e-10, abs_tol=1e-15) and delta > 0, 'root contraction comparison differs')
    tilt = measurements['candidate_tilt']
    by_offset = {row['offset']: row for row in tilt['samples']}
    for slope in tilt['symmetric_slopes']:
        h = slope['step']
        measured = (by_offset[h]['point']['radius'] - by_offset[-h]['point']['radius']) / (2*h)
        require(math.isclose(measured, slope['radius'], rel_tol=1e-9, abs_tol=1e-15), 'radius slope not rederived')
    require(all(row['radius_difference'] < 0 for row in tilt['samples'] if row['offset'] < 0), 'negative offsets no longer contract more')
    conditional = receipt['conditional_candidate_tilt']
    gradient = conditional['determinant_gradient_source']['gradient_scaled']
    direction = conditional['direction_scaled']
    delta = conditional['fresh_normalization_source']['delta']
    prediction = -sum(g*n for g,n in zip(gradient,direction))/(2*math.sqrt(-delta))
    require(math.isclose(sum(n*n for n in direction), 1, rel_tol=1e-12), 'tilt direction not unit length')
    require(direction == tilt['direction'] and delta == tilt['center_point']['delta'], 'tilt inputs differ')
    require(math.isclose(prediction, conditional['predicted_directional_radius_slope'], rel_tol=1e-12), 'conditional slope not rederived')
    errors = [100*abs(slope['radius']-prediction)/abs(prediction) for slope in tilt['symmetric_slopes']]
    require(math.isclose(max(errors), conditional['maximum_percent_discrepancy'], rel_tol=1e-9), 'tilt discrepancy not rederived')
    source_outputs = {Path(row['path']).name: row['sha256'] for row in receipt['figures']['artifacts']}
    for name, expected in extension['figure_receipts'].items():
        data = (bundle/name).read_bytes()
        require(digest(data) == expected == manifest['receipts'][name], 'direct figure receipt drift')
        fr = json.loads(data)
        require(fr['data_source']['sha256'] == extension['source_receipts']['memory-connection-probe-receipt.json']['sha256'], 'direct figure source differs')
        require(fr['data_source']['audit_sha256'] == receipt['audit']['sha256'], 'direct figure full audit differs')
        require(fr['plotted_data'] == {k:measurements[k] for k in ('candidate_tilt','before_after_axes','trace_roots')}, 'direct plotted data differs')
        require(fr['alt_text'] in html.unescape((bundle/'index.md').read_text()), 'direct figure alt differs')
        provenance = fr['provenance']
        require(provenance['visual_review'] == receipt['figures']['final_visual_review'], 'direct visual review differs')
        if repo:
            require(digest(git_show(repo, provenance['generator_commit'], provenance['generator'])) == provenance['generator_sha256'], 'direct generator differs')
        for output, expected_output in provenance['outputs'].items():
            source_name = output.replace('fig-memory-direct-probes','connection-probe')
            require(source_outputs[source_name] == expected_output, 'direct copied figure differs from audit')
            require(digest((bundle/output).read_bytes()) == expected_output, 'direct figure bytes drift')
    require(digest(Path(__file__).read_bytes()) == extension['verifier_sha256'], 'direct verifier hash drift')
    return len(extension['new_numbers']), len(extension['figure_receipts'])



def verify_minima(bundle, manifest, prose, repo):
    extension = next(e for e in manifest['extensions'] if e['id'] == 'memory-minima-20261022')
    receipts = {}
    for name, meta in extension['source_receipts'].items():
        data = (bundle/name).read_bytes()
        require(digest(data) == meta['sha256'] == manifest['receipts'][name], 'minima receipt hash drift')
        if repo:
            require(data == git_show(repo, meta['commit'], meta['source_path']), 'minima receipt differs from Git')
        receipts[name] = json.loads(data)
    receipt = receipts['contraction-minima-receipt.json']; plan = receipts['contraction-minima-plan.json']
    require(receipt['complete'] and receipt['audit_finished'] and not receipt['untested_or_incomplete_stages']
            and receipt['terminal_failure'] is None, 'minima audit incomplete')
    require(receipt['plan_sha256'] == digest((bundle/'contraction-minima-plan.json').read_bytes()), 'minima plan binding differs')
    counts, validation, sections = receipt['counts'], receipt['validation'], receipt['sections']
    cfg = plan['contraction_minima']
    require(counts['point_attempts'] == counts['qualified_points'] == 223 and counts['unqualified_or_incomplete_points'] == 0,
            'point qualification count differs from completed pilot')
    require(len(sections) == counts['completed_sections'] == 12 and all(s['complete'] for s in sections), 'section completion differs')
    require(sum(s['evaluation_count'] for s in sections) == 195, 'section callback count differs')
    require([s['status'] for s in sections] == ['converged']*8+['no_bracket','converged','no_bracket','converged'],
            'section result pattern differs')
    require(all(len(s['minima']) == 1 and s['minima'][0]['s'] == 0 for s in sections[:8]), 'outer/middle minimum pattern differs')
    require(sections[8]['mesh_monotonicity'] == 'strictly decreasing'
            and sections[10]['mesh_monotonicity'] == 'strictly increasing'
            and all(s['evaluation_count'] == len(cfg['section_mesh']) and not s['minima'] for s in (sections[8],sections[10])),
            'no-bracket mesh result differs')
    threshold = validation['empirical_radius_floor']*cfg['signal_multiple']
    require(threshold == validation['finite_contrast_threshold'], 'operational contrast threshold differs')
    minima, contrasts = [], []
    for section in sections:
        for minimum in section['minima']:
            minima.append(minimum)
            require(minimum['beta'] == section['beta'] == receipt['center']['beta'], 'fixed memory setting differs')
            width = minimum['refined_bracket'][1]-minimum['refined_bracket'][0]
            require(width == minimum['bracket_width'] and 0 < width <= cfg['coordinate_tolerance'], 'measured bracket width differs')
            require(minimum['refined_bracket'][0] <= minimum['s'] <= minimum['refined_bracket'][1], 'candidate outside measured bracket')
            require([v['offset'] for v in minimum['validation']] == cfg['validation_offsets'], 'finite comparison scales differ')
            for check in minimum['validation']:
                require(check['status'] == 'complete' and check['resolved'] and not check['lower_neighbor_resolved'], 'finite contrast gate incomplete')
                require(len(check['samples']) == 2, 'two actual neighbors required')
                actual = [s['radius']-minimum['radius'] for s in check['samples']]
                require(actual == check['contrasts'] and min(actual) > threshold, 'finite contrasts not rederived')
                for sign, sample in zip((-1,1),check['samples']):
                    require(sample['s'] == minimum['s']+sign*check['offset']
                            and cfg['section_mesh'][0] <= sample['s'] <= cfg['section_mesh'][-1], 'finite neighbor misplaced')
                contrasts.extend(actual)
    require(len(minima) == counts['sections_with_qualified_minimum'] == 10
            and len(contrasts) == counts['finite_validation_contrasts'] == 40, 'minimum/contrast count differs')
    require(min(contrasts) == validation['minimum_passed_finite_contrast']
            and min(contrasts) > 5*threshold and validation['maximum_full_state_closure'] < 1e-12,
            'stated contrast or closure bound fails')
    require(all(s['minima'][0]['s'] != 0 and s['minima'][0]['tau'] != 0 for s in (sections[9],sections[11])),
            'inner displaced/nonzero-trace claim differs')
    for entry in extension['new_numbers']:
        require(entry in manifest['numbers'] and entry['receipt_sha256'] == extension['source_receipts'][entry['receipt']]['sha256'],
                'minima number binding differs')
        expected = rendered(field(receipts[entry['receipt']], entry['field']), entry)
        require(expected in normal(entry['appears_as']) and normal(entry['appears_as']) in prose,
                'minima numerical display not rederived: '+entry['id'])
    for name, sha in extension['figure_receipts'].items():
        content = (bundle/name).read_bytes()
        require(digest(content) == sha == manifest['receipts'][name], 'minima figure receipt drift')
        figure = json.loads(content)
        require(figure['figure_id'] == 'M4' and 'Figure M4:' in prose, 'minima figure label differs')
        require(figure['data_source']['sha256'] == extension['source_receipts']['contraction-minima-receipt.json']['sha256']
                and figure['data_source']['audit_sha256'] == receipt['audit']['sha256'], 'minima figure data binding differs')
        require(figure['plotted_section_summaries'] == sections, 'minima figure summaries differ')
        require(figure['alt_text'] in html.unescape((bundle/'index.md').read_text()), 'minima alt text differs')
        provenance = figure['provenance']
        source_files = {Path(x['path']).name:x['sha256'] for x in receipt['figures']['files']}
        for companion in receipt.get('figure_companions',[]):
            source_files.update({Path(x['path']).name:x['sha256'] for x in companion['files']})
        for local, source_name in provenance['copied_source_names'].items():
            require(provenance['outputs'][local] == source_files[source_name], 'copied minima figure differs from public receipt')
        publication = provenance['publication']
        prod, review = publication['production_receipt'], publication['visual_review']
        require(prod['receipt_sha256'] == receipt['audit']['sha256'] and prod['scientific_complete'] is True
                and review['passed'] is True, 'publication source audit/review differs')
        for kind in ('production', 'visual_review'):
            prefix = kind+'_receipt' if kind == 'production' else kind
            file_key = prefix+'_file'; sha_key = prefix+'_sha256'
            content = (bundle/publication[file_key]).read_bytes()
            require(digest(content) == publication[sha_key] and json.loads(content) == publication[prefix],
                    'publication copied receipt/review drift')
        require(review['rendering_receipt']['sha256'] == publication['production_receipt_sha256'],
                'publication review binds a different rendering')
        pub_files = {x['path']:x['sha256'] for x in prod['figures']}
        for file_extension in ('png','pdf'):
            require(provenance['outputs']['fig-memory-section-minima.'+file_extension]
                    == pub_files['section-minima-center-zoom-publication.'+file_extension], 'publication image bytes differ')
        semantics = prod['caption_semantics']
        require(semantics['coordinate_scales'] == plan['coordinate_scales']
                and semantics['planned_longitudinal_distance'] == min(cfg['longitudinal_steps'])
                and semantics['display_multiplier'] == 1e4 and r'10^{-4}' in prose,
                'publication caption coordinate semantics differ')
        if repo:
            for path, expected in prod['source_sha256'].items():
                require(digest(git_show(repo,prod['source_commit'],path)) == expected, 'publication source differs from Git')
        mobile = provenance['mobile']
        require(mobile['production_receipt']['receipt_sha256'] == receipt['audit']['sha256']
                and mobile['visual_review']['passed'] is True, 'mobile source audit/review differs')
        require(digest((bundle/mobile['production_receipt_file']).read_bytes()) == mobile['production_receipt_sha256']
                and json.loads((bundle/mobile['production_receipt_file']).read_text()) == mobile['production_receipt'], 'mobile receipt drift')
        require(digest((bundle/mobile['visual_review_file']).read_bytes()) == mobile['visual_review_sha256']
                and json.loads((bundle/mobile['visual_review_file']).read_text()) == mobile['visual_review'], 'mobile review drift')
        mobile_files = {x['path']:x['sha256'] for x in mobile['production_receipt']['figures']}
        require(provenance['outputs']['fig-memory-section-minima-mobile.png'] == mobile_files['section-minima-center-zoom-mobile.png'],
                'mobile image differs from audited rendering')
        if repo:
            for path, expected in mobile['production_receipt']['source_sha256'].items():
                require(digest(git_show(repo,mobile['production_receipt']['source_commit'],path)) == expected, 'mobile source differs from Git')
        for output, expected in provenance['outputs'].items():
            require(digest((bundle/output).read_bytes()) == expected, 'minima image/PDF bytes drift')
    require(digest(Path(__file__).read_bytes()) == extension['verifier_sha256'], 'minima verifier hash drift')
    return len(extension['new_numbers']), len(extension['figure_receipts'])



def mixed_display_rows(receipt):
    return [{'case_index':c['case_index'], 'beta':c['beta'], 'beta_offset':c['beta_offset'],
             'status':c['status'], 'feature':c['f_minus']['value'],
             'feature_error':c['f_minus']['empirical_total_error'], 'feature_status':c['f_minus']['status'],
             'primary_tau':c['decisive_iteration']['coarse_corrector']['equations'][0],
             'cyclic_traces':c['decisive_iteration']['cyclic_measurements']['phases']}
            for c in receipt['cases']]


def verify_mixed(bundle, manifest, prose, repo):
    extension = next(e for e in manifest['extensions'] if e['id'] == 'memory-mixed-20261022')
    receipts = {}
    for name, meta in extension['source_receipts'].items():
        content = (bundle/name).read_bytes()
        require(digest(content) == meta['sha256'] == manifest['receipts'][name], 'mixed receipt hash drift')
        if repo:
            require(content == git_show(repo,meta['commit'],meta['source_path']), 'mixed receipt differs from Git')
        receipts[name] = json.loads(content)
    r, plan = receipts['mixed-junction-receipt.json'], receipts['mixed-junction-plan.json']
    cfg = plan['mixed_junction']; cases = r['cases']; life = r['lifecycle']['audit_lifecycle']
    require(r['audit_finished'] and r['complete'] and r['execution_finished'] and not r['untested_or_incomplete_stages']
            and r['terminal_failure'] is None and not r['failed_or_incomplete_point_admissions']
            and not r['unqualified_or_incomplete_points'], 'mixed result has incomplete or failed evidence')
    require(life['python_finalization'] and life['python_complete'] and life['controller']['worker_reaped']
            and life['controller']['execution_complete'] and life['controller']['scientific_complete']
            and not life['controller']['timed_out'] and life['controller']['exit_code'] == 0, 'mixed lifecycle incomplete')
    require(r['plan_sha256'] == digest((bundle/'mixed-junction-plan.json').read_bytes())
            and cfg == r['declared_case_and_precision_settings'], 'mixed frozen plan binding differs')
    counts = r['counts']
    require(counts['point_attempts'] == counts['orbit_spectral_qualified_points'] and counts['unqualified_or_incomplete_points'] == 0
            and counts['planned_cases'] == counts['attempted_cases'] == counts['admitted_cases'] == len(cases) == len(cfg['beta_offsets'])
            and counts['unattempted_cases'] == 0, 'mixed count/completion mismatch')
    require(r['scientific_metadata']['period'] == 18 and cfg['primary_step'] == cfg['stencil_steps'][0]
            and cfg['stencil_steps'] == [.0003,.0001], 'mixed field/period design differs')
    labels = []
    for index, (c,offset) in enumerate(zip(cases,cfg['beta_offsets'])):
        require(c['case_index'] == index and c['beta_offset'] == offset
                and c['beta'] == cfg['target']['beta']+offset and c['status'] == 'admitted' and c['complete'], 'mixed case identity/status differs')
        d = c['decisive_iteration']; a = d['admission']; u = d['uncertainty']; g = d['coarse_conditional_geometry']
        k = d['coarse_corrector']; cyclic = d['cyclic_measurements']; b = u['branch_envelopes']; inter = u['intersection']
        require(d['center'] == c['final_center'] == cyclic['coordinate'] and d['beta'] == c['beta'] == cyclic['beta']
                and c['final_point_index'] == cyclic['point_index'] and [x['phase'] for x in cyclic['phases']] == [0,6,12], 'mixed corrected center/cyclic phase binding differs')
        require(d['coarse_stencil']['center_point_index'] == d['fine_stencil']['center_point_index'], 'mixed matched stencils use different centers')
        for prefix,step in zip(('coarse','fine'),cfg['stencil_steps']):
            stencil, geometry = d[prefix+'_stencil'], d[prefix+'_conditional_geometry']
            require(stencil['step'] == step and stencil['fields']['tau']['value'] == geometry['input_tau']
                    and stencil['fields']['delta']['value'] == geometry['input_delta'], 'mixed raw field/geometry drift')
            p = geometry['projection']
            require(p['input_tau'] == geometry['input_tau'] and p['absolute_trace_change'] == abs(p['input_tau'])
                    and p['enforced_tau'] == 0 and p['spatial_displacement_applied'] is False
                    and p['kind'] == 'conditional trace-value projection at the same coordinate', 'mixed projection silently changed measurement')
        require(k['equations'][0] == g['input_tau'] and k['equations'][1] == g['branches']['positive_tau_side']['feature']
                and k['f_minus'] == g['branches']['negative_tau_side']['feature'], 'mixed feature equations differ')
        checks = {
            'actual_cyclic_trace':max(abs(p['tau']) for p in cyclic['phases']) <= cfg['trace_tolerance'],
            'actual_primary_trace':abs(g['input_tau']) <= cfg['trace_tolerance'],
            'f_plus_residual':abs(k['equations'][1]) <= cfg['f_plus_tolerance'],
            'coordinate_correction':k['correction_norm'] <= cfg['coordinate_tolerance'],
            'projection_displacement':g['projection']['predicted_displacement_norm'] <= cfg['maximum_projection_displacement'],
            'paired_field_jets':all(v <= cfg['jet_difference_limits'][name][key] for name,ds in u['paired_jet_differences'].items() for key,v in ds.items()),
            'regular_crease_with_uncertainty':u['cusp']['trace_gradient_lower'] >= cfg['minimum_trace_gradient'],
            'both_normalized_jacobians':min(u['normalized_jacobian_minimum_singular_values']) >= cfg['minimum_normalized_jacobian_singular_value'],
            'strict_cusp_slopes':u['cusp']['left_upper'] < 0 < u['cusp']['right_lower'],
            'positive_simple_transported_normals':all(x['positive_curvature_lower'] > 0 and x['gap_lower'] > 0
                and x['transport_overlap_lower'] >= cfg['minimum_reference_overlap'] for x in b.values()),
            'feature_precision':all(x['feature_error'] <= cfg['maximum_feature_uncertainty'] for x in b.values()),
            'intersection_precision':inter['location_error_norm'] <= cfg['maximum_intersection_location_uncertainty'],
            'f_minus_total_precision':inter['f_minus_total_error'] <= cfg['maximum_f_minus_total_uncertainty']}
        require(checks == a['checks'] and len(checks) == 13 and all(checks.values()) and a['passed'], 'mixed admission gates not rederived')
        require(math.isclose(inter['f_minus_location_transfer'],inter['f_minus_nominal_transfer']+inter['inverse_rotation_transfer']+inter['f_minus_gradient_transfer'],rel_tol=1e-14,abs_tol=0)
                and inter['f_minus_total_error'] == b['negative_tau_side']['feature_error']+inter['f_minus_location_transfer'], 'mixed empirical envelope components differ')
        value,error = k['f_minus'],inter['f_minus_total_error']
        label = ('resolved_positive' if value > cfg['signal_multiple']*error else 'resolved_negative' if value < -cfg['signal_multiple']*error
                 else 'residual_qualified_near_zero' if abs(value) <= cfg['f_minus_tolerance'] and error <= cfg['maximum_f_minus_total_uncertainty']
                 else 'unresolved_sign')
        for fm in (c['f_minus'],a['f_minus']):
            require(fm['value'] == value and fm['empirical_total_error'] == error and fm['status'] == label, 'mixed sign/admission differs')
        labels.append(label)
        if index:
            parent = c['predictor_parent']; central = cases[0]
            require(parent['kind'] == 'fresh admitted central intersection' and parent['coordinate'] == central['final_center']
                    and parent['point_index'] == central['final_point_index'] and parent['beta'] == central['beta'], 'mixed outward predictor discarded central correction')
            expected = [x+offset*v for x,v in zip(central['final_center'],cfg['target']['moving_center_beta'])]
            require(all(abs(x-y) <= 1e-13 for x,y in zip(expected,c['predicted_center'])), 'mixed outward predictor coordinates differ')
    require(labels == ['residual_qualified_near_zero','resolved_positive','resolved_negative']
            and r['resolved_f_minus_sign_bracket'] is True, 'mixed published outcome is not supported')
    for e in extension['new_numbers']:
        require(e in manifest['numbers'] and e['receipt_sha256'] == extension['source_receipts'][e['receipt']]['sha256'], 'mixed number binding differs')
        expected = rendered(field(receipts[e['receipt']],e['field']),e)
        require(expected in normal(e['appears_as']) and normal(e['appears_as']) in prose, 'mixed number not rederived: '+e['id'])
    for name,expected in extension['figure_receipts'].items():
        content = (bundle/name).read_bytes()
        require(digest(content) == expected == manifest['receipts'][name], 'mixed figure receipt drift')
        fig = json.loads(content); p = fig['provenance']
        require(p['public_summary'] == receipts['mixed-junction-summary.json'], 'mixed public figure metadata differs from Git copy')
        require(fig['figure_id'] == 'M5' and 'Figure M5:' in prose and fig['alt_text'] in html.unescape((bundle/'index.md').read_text()), 'mixed figure label/alt mismatch')
        require(fig['data_source']['sha256'] == extension['source_receipts']['mixed-junction-receipt.json']['sha256']
                and fig['data_source']['audit_sha256'] == r['audit']['sha256'] and fig['plotted_case_summaries'] == mixed_display_rows(r), 'mixed figure evidence mismatch')
        for key in ('publication_production','publication_visual_review'):
            require(digest((bundle/p[key+'_file']).read_bytes()) == p[key+'_sha256']
                    and json.loads((bundle/p[key+'_file']).read_text()) == p[key], 'mixed publication receipt/review drift')
        for output,sha in p['outputs'].items():
            require(digest((bundle/output).read_bytes()) == sha, 'mixed image/PDF bytes drift')
        # Publication layout-specific audit/source/output bindings follow below.
        verify_mixed_publication(r,fig,repo)
    require(digest(Path(__file__).read_bytes()) == extension['verifier_sha256'], 'mixed verifier hash drift')
    return len(extension['new_numbers']),len(extension['figure_receipts'])


def verify_mixed_publication(receipt,figure,repo):
    p = figure['provenance']; production = p['publication_production']; review = p['publication_visual_review']; public = p['public_summary']
    require(production['scientific_complete'] is True and production['receipt_sha256'] == receipt['audit']['sha256']
            and public['full_audit'] == receipt['audit'] and public['numeric_receipt']['sha256'] == figure['data_source']['sha256'], 'mixed publication audit differs')
    require(public['figure_receipt']['sha256'] == p['publication_production_sha256']
            and public['visual_review']['sha256'] == p['publication_visual_review_sha256']
            and review['passed'] is True and review['figure_receipt'] == public['figure_receipt']
            and review['audit_receipt'] == receipt['audit'], 'mixed publication visual binding differs')
    require(production['source_commit'] == public['source_commit'] and production['source_hashes'] == public['source_hashes'], 'mixed publication sources differ')
    require(not production['unqualified_or_incomplete_points'] and not production['untested_or_incomplete_stages']
            and production['terminal_failure'] is None, 'mixed publication omitted failures')
    require(len(production['displayed_cases']) == len(receipt['cases']), 'mixed plotted case count differs')
    for row,c in zip(production['displayed_cases'],receipt['cases']):
        d = c['decisive_iteration']
        require(row['case_index'] == c['case_index'] and row['beta'] == c['beta'] and row['beta_offset'] == c['beta_offset']
                and row['admitted'] and row['status'] == c['status'] and row['feature_status'] == c['f_minus']['status']
                and row['feature'] == c['f_minus']['value'] and row['feature_error'] == c['f_minus']['empirical_total_error']
                and row['final_center'] == c['final_center'], 'mixed plotted feature or status differs')
        traces = [{'source':'primary center','phase':0,'tau':d['coarse_corrector']['equations'][0]}]
        traces += [{'source':'cyclic replay','phase':v['phase'],'tau':v['tau']} for v in d['cyclic_measurements']['phases']]
        require(row['actual_traces'] == traces, 'mixed plotted signed trace was altered')
    production_files = {x['path']:x['sha256'] for x in production['figures']}
    public_files = {Path(x['path']).name:x['sha256'] for x in public['external_figures']}
    original_files = {Path(x['path']).name:x['sha256'] for x in receipt['figures']['files']}
    for local,source in p['copied_source_names'].items():
        sha = p['outputs'][local]
        require(sha == (production_files.get(source) or original_files.get(source)), 'mixed figure copy differs from source')
        if source in production_files:
            require(sha == public_files[source] and any(x['sha256'] == sha for x in review['reviewed']), 'mixed figure lacks frozen visual review')
    if repo:
        for path,sha in production['source_hashes'].items():
            require(digest(git_show(repo,production['source_commit'],path)) == sha, 'mixed generator differs from frozen source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', required=True)
    parser.add_argument('--research-repo', type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'provenance.json').read_text())
    extension = next(e for e in manifest['extensions'] if e['id'] == 'memory-20261021')
    prose = normal((bundle / 'index.md').read_text())
    receipts = {}
    for name, meta in extension['source_receipts'].items():
        data = (bundle / name).read_bytes()
        require(digest(data) == meta['sha256'] == manifest['receipts'][name], 'receipt hash drift: ' + name)
        if args.research_repo:
            require(data == git_show(args.research_repo, meta['commit'], meta['source_path']), 'receipt differs from Git source: ' + name)
        receipts[name] = json.loads(data)
    for entry in extension['new_numbers']:
        require(entry in manifest['numbers'], 'extension entry absent from main number manifest')
        require(entry['receipt_sha256'] == extension['source_receipts'][entry['receipt']]['sha256'], 'number receipt hash drift')
        value = field(receipts[entry['receipt']], entry['field'])
        expected = rendered(value, entry)
        require(expected in normal(entry['appears_as']), 'declared display not rederived: ' + entry['id'] + ' expected ' + expected)
        require(normal(entry['appears_as']) in prose, 'number absent from post: ' + entry['id'])
    local = receipts['memory-local-extension-receipt.json']
    parent = receipts['memory-continuation-receipt.json']
    require(local['complete'] and local['counts']['unqualified_or_incomplete_points'] == 0, 'local audit incomplete')
    require(not parent['complete'] and parent['rejected_corrector_steps'], 'historical failed attempt must remain failed')
    require(local['combined_orbit_continuity']['summary']['all_aligned_token_cycles_match'], 'token continuity assertion failed')
    low, high = local['height_root']['initial_sign_bracket']
    require(low['tau'] > 0 > high['tau'], 'sign bracket lost')
    require(0 < local['height_root']['tau'] <= local['height_root']['height_tolerance'], 'residual-qualified positive candidate changed')
    scan = receipts['memory-junction-scan-receipt.json']
    cells = math.prod(scan['scan_settings']['resolution'])
    period = scan['map_equivalence']['period']
    require(scan['status'].startswith('complete;'), 'scan audit incomplete')
    require(scan['totals']['cells'] == cells and scan['totals']['period_counts'] == {str(period): cells}, 'scan recurrence count differs')
    require(scan['reference']['totals']['cells'] == cells, 'reference cell count differs')
    require(scan['totals']['lyapunov_range'][0] <= scan['totals']['lyapunov_range'][1] < 0, 'scan growth range is not wholly negative')
    readout = scan['readout_comparison']
    require(readout['same_canonical_six_token_readout'] and readout['cells_in_each_panel'] == cells, 'readout comparison differs')
    require(primitive_token_period(readout['token_ids']) == readout['token_period'], 'canonical readout period differs')
    require(scan['root_evidence']['beta'] == local['height_root']['parameter'], 'scan memory candidate differs from continuation')
    require(scan['plot']['centers'] == [scan['reference']['center'], scan['root_evidence']['center']], 'translated scan centers differ')
    require(scan['plot']['interpolation'] == 'nearest' and not scan['plot']['overlaid_fitted_curves'], 'image rendering scope differs')
    require(all(not c['center_is_measured_cell'] and all(not a['contains_exact_center'] for a in c['axes']) for c in scan['plot']['center_sampling']), 'center sampling qualification differs')
    for name, expected in extension['figure_receipts'].items():
        data = (bundle / name).read_bytes()
        require(digest(data) == expected == manifest['receipts'][name], 'figure receipt drift: ' + name)
        fr = json.loads(data)
        if name == 'fig-memory-continuation.receipt.json':
            points = sorted(parent['accepted_centers'] + local['accepted_centers'], key=lambda p: p['parameter'])
            plot = [{'beta': p['parameter'], 'memory_percent': 100*p['parameter'], 'tau': p['tau'], 'scaled_center': p['center']} for p in points]
            require(plot == fr['plotted_data'] and len(plot) == local['figures']['accepted_samples'], 'plotted data differs')
            require(fr['highlighted_root_beta'] == local['height_root']['parameter'], 'root marker differs')
            require(fr['alt_text'] in html.unescape((bundle / 'index.md').read_text()), 'figure alt text differs')
        elif name == 'fig-memory-junction-pair.receipt.json':
            require(fr['figure_id'] == 'M2' and 'Figure M2:' in prose, 'scan figure label differs')
            require(fr['plot'] == scan['plot'], 'scan figure plot settings differ')
            require(fr['alt_text'] in html.unescape((bundle / 'index.md').read_text()), 'scan figure alt text differs')
            require(fr['data_source']['sha256'] == extension['source_receipts']['memory-junction-scan-receipt.json']['sha256'], 'scan figure source receipt differs')
            source_outputs = {Path(o['path']).name: o['sha256'] for o in scan['figures']}
            require(all(source_outputs.get(n) == h for n,h in fr['provenance']['outputs'].items()), 'scan figure not bound to public audit')
            require(fr['provenance']['visual_review'] == scan['visual_review'], 'scan source visual review differs')
        if args.research_repo and 'generator_commit' in fr['provenance']:
            p = fr['provenance']
            require(digest(git_show(args.research_repo, p['generator_commit'], p['generator'])) == p['generator_sha256'], 'generator source differs')
        generator = bundle / fr['provenance']['generator']
        if generator.is_file():
            require(digest(generator.read_bytes()) == fr['provenance']['generator_sha256'], 'local figure generator drift')
        for output, expected_output in fr['provenance']['outputs'].items():
            require(digest((bundle / output).read_bytes()) == expected_output, 'figure bytes drift: ' + output)
    if 'verifier_sha256' in extension:
        require(digest(Path(__file__).read_bytes()) == extension['verifier_sha256'], 'verifier hash drift')
    if args.research_repo:
        # Check every newly introduced research citation in the memory subsection,
        # plus the pinned receipt records. Historic arxiv-v1 URLs are unchanged.
        section = (bundle / 'index.md').read_text().split('{#memory-third-knob}', 1)[1].split('### Symbolic dynamics', 1)[0]
        links = set(re.findall(r'https://github\.com/tdj28/neural-shrimp/blob/([0-9a-f]{40})/([^\s)"<>]+)', section))
        for commit, path in links:
            git_show(args.research_repo, commit, path)
    direct_numbers, direct_figures = verify_direct(bundle, manifest, prose, args.research_repo)
    minima_numbers, minima_figures = verify_minima(bundle, manifest, prose, args.research_repo)
    mixed_numbers, mixed_figures = verify_mixed(bundle, manifest, prose, args.research_repo)
    print(f'Memory extensions verified: {len(extension["new_numbers"]) + direct_numbers + minima_numbers + mixed_numbers} numerical entries; {len(extension["figure_receipts"]) + direct_figures + minima_figures + mixed_figures} figure receipts; source Git checks ' + ('passed.' if args.research_repo else 'not requested.'))


if __name__ == '__main__':
    main()
