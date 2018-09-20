from time import time
import pickle
import numpy as np
from math import *
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter
from gatspy.periodic import RRLyraeTemplateModeler, LombScargleFast, LombScargle
import gatspy.datasets as datasets
from ftperiodogram.modeler import FastTemplatePeriodogram, SlowTemplatePeriodogram
from ftperiodogram.template import Template
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from collections import namedtuple

plt.style.use('./ggplot-like.mplstyle')
rc('text', usetex=False)
rc('font', **{'family' : 'sans-serif', 'sans-serif' : [ 'cmss10' ]})


default_settings = dict(
        wspace = 0.02,
        left = 0.12,
        bottom = 0.12,
        right = 0.9,
        top = 0.9,
        locator_axtmp = 0.5,
        locator_axpdg = None,
        ratio_pdglen_tmplen = 5,
        keep_grid_sizes_equal = True,
        title_pad = 0.01,
        label_fontsize = 12,
        title_fontsize = 12,
        tick_fontsize = 10,
        annotation_fontsize = 12,
        font_color = '#555555',
        colorfunc_a = 0.8,
        colorfunc_b = 0.2,
        answer_color = 'r',
        annotate_x0 = -0.06,
        axsize = 5,
        tmp_height_frac = 0.6,
        data_params = dict(fmt='o', ecolor='0.6',
             markeredgecolor='none', markersize=3,
             markerfacecolor='k', capsize=0, linewidth=1),

    )

rms = lambda y : sqrt(np.mean(np.power(y, 2)))
rror = lambda ym, y : rms(y - ym) / rms(y)

def get_boundaries_for_axtmp_and_axpdg(maxfrq, settings):
    possible_gridsizes = [ 1, 2, 5, 10, 15, 20, 25, 50, 100 ]

    dx_all = settings['right'] - settings['left']
    ltmp, lpdg = None, None
    if settings['keep_grid_sizes_equal']:
        ratio = lambda g2 : settings['locator_axtmp'] * int(maxfrq / g2)
        i = np.argmin([ abs(ratio(g2) - settings['ratio_pdglen_tmplen'])
                                for g2 in possible_gridsizes ])

        settings['locator_axpdg'] = possible_gridsizes[i]

        ltmp = (dx_all - settings['wspace']) / (1 + ratio(settings['locator_axpdg']))
    else:
        ltmp = (dx_all - settings['wspace']) / (1 + settings['ratio_pdglen_tmplen'])

    lpdg = settings['right'] - settings['left'] - settings['wspace'] - ltmp


    bounds_ax_tmp = (   settings['left'],
                        settings['bottom'],
                        ltmp,
                        settings['top'] - settings['bottom']
                    )
    bounds_ax_pdg = ( settings['left'] + bounds_ax_tmp[2] + settings['wspace'],
                      settings['bottom'],
                      settings['right'] - settings['left'] \
                           - bounds_ax_tmp[2] - 0.5 * settings['wspace'],
                      settings['top'] - settings['bottom'] )

    return bounds_ax_tmp, bounds_ax_pdg


def flatten(iterable):
    out = []
    for i in iterable:
        if hasattr(i,'__iter__'):
            out.extend(flatten(i))
        else:
            out.append(i)
    return out

def generate_random_signal(n, sigma=1.0, freq=1.0, template=None):
    x = np.sort(np.random.rand(n))
    dy = sigma * np.random.normal(size=n, loc=0)
    err = np.ones_like(x) * sigma

    y = np.cos(freq * x)
    if not template is None:
        y = template(x * freq) + dy

    return x, y, err


def adjust_figure(f, settings):
    keywords = ['left', 'right', 'top', 'bottom', 'hspace', 'wspace']
    keywords = list(filter(lambda a : a in settings, keywords))

    if len(keywords) > 0:
        kwargs = { key : settings[key] for key in keywords }
        f.subplots_adjust(**kwargs)

def translate_color(ax, settings):
    for key, value in settings.iteritems():
        if isinstance(value, dict):
            settings[key] = translate_color(ax, value)

        if value == 'axes_background':
            settings[key] = ax.get_facecolor()
    return settings

def clean_up_figure(f, settings):

    # save
    if not settings['fname'] is None:
        fname = os.path.join(settings['plot_dir'], settings['fname'])
        #f.savefig(fname, dpi=settings['dpi'])
        f.savefig(fname)

def clean_up_axis(ax, settings):

    translate_color(ax, settings)

    # format x tick labels
    [ label.set_fontsize(settings['tick_fontsize']) and \
      label.set_fontfamily('sans-serif') \
        for label in ax.get_xticklabels()]

    # format y tick labels
    [ label.set_fontsize(settings['tick_fontsize']) and \
      label.set_fontfamily('sans-serif') \
        for label in ax.get_yticklabels()]

    # remove ticks
    if settings['remove_ticks']:
        ax.tick_params(axis='both', which='both', length=0)

    # make sure legend font
    legend = ax.get_legend()
    if not legend is None:
        for txt in legend.get_texts():
            txt.set_color(settings['font_color'])
            txt.set_family('sans-serif')
            txt.set_fontsize(settings['legend_fontsize'])

        legend.set_frame_on(settings['legend_frameon'])

        legend.get_frame().set_facecolor(settings['legend_facecolor'])
        legend.get_frame().set_edgecolor(settings['legend_edgecolor'])

    ax.set_xlabel(ax.get_xlabel(), color=settings['font_color'],
                     fontsize=settings['label_fontsize'])
    ax.set_ylabel(ax.get_ylabel(), color=settings['font_color'],
                     fontsize=settings['label_fontsize'])


def open_results(name, settings, mode):
    fname = os.path.join(settings['results_dir'], settings[name])
    return open(fname, mode)



def get_default_template(nharmonics=5):
    rrl_templates = datasets.rrlyrae.fetch_rrlyrae_templates()
    xt, yt = rrl_templates.get_template('100r')

    template = Template.from_sampled(yt, nharmonics=nharmonics)
    return template


def get_timing_vs_nharmonics(x, y, yerr, hvals, filename=None, overwrite=True,
                                only_use_saved_data = False):

    # load old results
    results = {}
    if not filename is None and os.path.exists(filename):
        old_results = pickle.load(open(filename, 'rb'))
        results.update(old_results)

    # return if nothing to do
    if all([ h in results for h in hvals ]):
        return hvals, [ results[h] for h in hvals ]

    if only_use_saved_data:
        return zip(*[ (h, results[h]) for h in hvals if h in results ])

    for h in hvals:
        if h in results:
            continue

        print("   H = ", h)

        template = get_default_template(nharmonics=h)
        #template.precompute()

        model = FastTemplatePeriodogram(template=template)
        model.fit(x, y, yerr)

        t0 = time()
        model.autopower()
        results[h] = time() - t0

        print("   %.4f seconds"%(results[h]))
        if not filename is None and overwrite:
            pickle.dump(results, open(filename, 'wb'))

    return hvals, [ results[h] for h in hvals ]

def get_timing_lombscargle(n, ofac, hfac):

    x, y, dy = generate_random_signal(n)

    Nf = int(floor(0.5 * len(x) * ofac * hfac))
    df = 1./(ofac * (max(x) - min(x)))
    f0 = df

    model = LombScargleFast(silence_warnings=True)
    model.fit(x, y, dy)
    t0 = time()
    model.score_frequency_grid(f0, df, Nf)
    dt = time() - t0

    return dt

def sort_timing_dict(tdict, col='ndata'):
    keys = tdict.keys()
    inds = np.argsort(tdict[col])
    for key in keys:
        tdict[key] = [ tdict[key][i] for i in inds ]
    return tdict

def select_from_dict(tdict, values, col='ndata'):
    tcopy = {}
    tcopy.update(tdict)

    inds = [ i for i in range(len(tcopy[col])) if tcopy[col][i] in values ]

    for key in tcopy:
        tcopy[key] = [ tcopy[key][i] for i in inds ]

    return tcopy

def get_timing_vs_ndata(nvals, nharmonics, filename=None, overwrite=True,
                        time_gatspy=True, only_use_saved_data=False, time_lomb_scargle=False):
    #if template is None:
    template = get_default_template(nharmonics=nharmonics)
    #template.precompute()

    # load saved results
    results = {}
    if not filename is None and os.path.exists(filename):
        old_results = pickle.load(open(filename, 'rb'))
        results.update(old_results)
    else:
        results = { name : [] for name in [ 'nfreqs', 'ndata', 'tftp', 'tgats' ]}

    if only_use_saved_data:
        return select_from_dict(results, nvals)

    # return if nothing to do
    if all([ n in results for n in nvals ]):
        return [ results[n] for n in nvals ]

    for n in nvals:
        if n in results['ndata']:
            continue

        results['ndata'].append(n)

        x, y, dy = generate_random_signal(n)

        # time FTP
        print("timing: n = %d, h = %d, ftp"%(n, nharmonics))

        model = FastTemplatePeriodogram(template=template)
        model.fit(x, y, dy)

        t0 = time()
        frq, p = model.autopower()
        results['tftp'].append( time() - t0 )

        print("   done in %.4f seconds"%(results['tftp'][-1]))
        results['nfreqs'].append(len(frq))

        if time_gatspy:
            # time GATSPY
            print("timing: n = %d, gatspy"%(n))

            model = SlowTemplatePeriodogram(template=template)
            model.fit(x, y, dy)

            t0 = time()
            p = model.power(frq)
            results['tgats'].append(time() - t0)

            print("   done in %.4f seconds"%(results['tgats'][-1]))
        else:
            results['tgats'].append(-1)


        # SORT results
        results = sort_timing_dict(results)

        # save
        if not filename is None and overwrite:
            pickle.dump(results, open(filename, 'wb'))

    return select_from_dict(results, nvals)

def get_timing_vs_ndata_at_const_nfreq(nvals, nharmonics, max_freq, filename=None, overwrite=True, time_slow=True):

    #if template is None:
    template = get_default_template(nharmonics=nharmonics)
    template.precompute()

    # load saved results
    results = {}
    if filename is None:
        filename = './saved_results/timing_results_nh%d_maxfrq%.1f.pkl'%(nharmonics, max_freq)
    if not filename is None and os.path.exists(filename):
        old_results = pickle.load(open(filename, 'rb'))
        results.update(old_results)


    for n in nvals:
        if n in results:
            continue

        x, y, dy = generate_random_signal(n)
        x[0] = 0
        x[-1] = 1

        # time FTP
        print("timing: n = %d, h = %d, ftp"%(n, nharmonics))

        model = FastTemplatePeriodogram(template=template)
        model.fit(x, y, dy)

        t0 = time()
        frq, p = model.autopower(maximum_frequency=max_freq)
        tftp = time() - t0

        print("   done in %.4f seconds"%(tftp))

        if time_slow:
            print("timing: n = %d, h = %d, slow"%(n, nharmonics))
            model = SlowTemplatePeriodogram(template=template)
            model.fit(x, y, dy)

            t0 = time()
            p = model.power(frq)
            tslow = time() - t0

            print("   done in %.4f seconds"%(tslow))
        else:
            tslow = -1

        results[n] = (tftp, tslow)

        # save
        if not filename is None and overwrite:
            pickle.dump(results, open(filename, 'wb'))

    return zip(*[ results[n] for n in nvals ])

HVAL_WITH_GATSPY_TIMING_DATA = 6
def plot_timing_vs_ndata(settings=default_settings):

    f, ax = plt.subplots(figsize=(settings['axsize'], settings['axsize']))

    settings = translate_color(ax, settings)

    fname = os.path.join(settings['results_dir'], settings['timing_filename'])

    nharms = settings['nharmonics']
    if not hasattr(nharms, '__iter__'):
        nharms = [ nharms ]

    tls = None
    if settings['plot_lomb_scargle']:
        tls = []
        for n in settings['ndata']:
            tls.append(get_timing_lombscargle(n, 10, 3))


    for i, h in enumerate(nharms):

        time_gatspy = (h == HVAL_WITH_GATSPY_TIMING_DATA)
        filename = fname.replace('.pkl', '_h%d.pkl'%(h))
        timing_data =  get_timing_vs_ndata(settings['ndata'], h, filename=filename,
                                                            time_gatspy=time_gatspy,
                                                            only_use_saved_data=settings['only_use_saved_data'])

        ndata  = np.array(timing_data['ndata'])
        nfreqs = np.array(timing_data['nfreqs'])
        tftp   = np.array(timing_data['tftp'])
        tgats  = np.array(timing_data['tgats'])



        label = None if h < max(nharms) else 'Fast template periodogram'
        color = settings['scatter_params_ftp']['c']
        lw    = settings['linewidth']
        spars = {}
        spars.update(settings['scatter_params_ftp'])
        fudge = 0.7
        ls = '-' if h == max(nharms) else ':'
        #ls = '-'
        q = float(h - min(nharms)) / float(max(nharms) - min(nharms))

        spars['alpha'] = fudge * q + (1 - fudge)

        ax.scatter(ndata, tftp, label=label, **spars)
        ax.plot(ndata, tftp, color=color, lw=lw, alpha=spars['alpha'], ls=ls)

        # now label this

        xoffset = 0.08 * (settings['xlim'][1] - settings['xlim'][0])

        ax.text(ndata[-1] + xoffset, tftp[-1], "$H = %d$"%(h), ha='left', va='center',
                   color=settings['font_color'], fontsize=settings['annotation_fontsize'],
                   bbox=settings['bbox'])

        if time_gatspy:

            ax.scatter(ndata, tgats,  label='Non-linear optimization', **settings['scatter_params_gatspy'])
            ax.plot(ndata, tgats, color=settings['scatter_params_gatspy']['c'],
                                                        lw=settings['linewidth'])

            if not tls is None:
                ax.scatter(ndata, tls, **settings['scatter_params_lomb_scargle'])
                ax.plot(ndata, tls, **settings['line_params_lomb_scargle'])
                ax.plot(ndata, tls, color=ax.get_facecolor(), lw=4, zorder=8)


    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel("Execution time [s]")


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Constant cadence')
    ax.set_xlim(*settings['xlim'])
    ax.set_ylim(*settings['ylim'])

    ax.legend(loc=settings['legend_loc'])

    adjust_figure(f, settings)
    clean_up_axis(ax, settings)
    clean_up_figure(f, settings)

def plot_timing_vs_ndata_const_freq(settings=default_settings):

    f, ax = plt.subplots(figsize=(settings['axsize'], settings['axsize']))

    settings = translate_color(ax, settings)

    #fname = os.path.join(settings['results_dir'], settings['timing_filename'])

    nharms = settings['nharmonics']
    if not hasattr(nharms, '__iter__'):
        nharms = [ nharms ]


    x, y, dy = generate_random_signal(10)
    x[0] = 0
    x[-1] = 1
    xoffset = 0.08 * (settings['xlim'][1] - settings['xlim'][0])
    nfrq = len(FastTemplatePeriodogram().fit(x, y, dy).autofrequency(maximum_frequency=settings['max_freq']))
    for i, h in enumerate(nharms):
        time_slow = (h==3)
        tftp, tslow =  get_timing_vs_ndata_at_const_nfreq(settings['ndata'], h,
                                        settings['max_freq'], time_slow=time_slow)


        label = None if h < max(nharms) else 'Fast template periodogram'
        color = settings['scatter_params_ftp']['c']
        lw    = settings['linewidth']
        spars = {}
        spars.update(settings['scatter_params_ftp'])
        fudge = 0.7
        ls = '-' if h == max(nharms) else ':'
        #ls = '-'
        q = 1.
        if len(nharms) > 1:
            q = float(h - min(nharms)) / float(max(nharms) - min(nharms))

        spars['alpha'] = fudge * q + (1 - fudge)

        ax.scatter(settings['ndata'], tftp, label=label, **spars)
        ax.plot(settings['ndata'], tftp, color=color, lw=lw, alpha=spars['alpha'], ls=ls)

        # now label this

        ax.text(settings['ndata'][-1] + xoffset, tftp[-1], "$H = %d$"%(h), ha='left', va='center',
                   color=settings['font_color'], fontsize=settings['annotation_fontsize'],
                   bbox=settings['bbox'])

        if time_slow:

            ax.scatter(settings['ndata'], tslow,  label='Non-linear optimization', **settings['scatter_params_nonlin'])
            ax.plot(settings['ndata'], tslow, color=settings['scatter_params_nonlin']['c'],
                                                        lw=settings['linewidth'])


    ax.text(0.05, 0.7, "$N_f=%d$"%(nfrq), ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('Number of datapoints')
    ax.set_ylabel("Execution time [s]")

    ax.set_title('Constant baseline')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(*settings['xlim'])
    ax.set_ylim(*settings['ylim'])

    ax.legend(loc=settings['legend_loc'])

    adjust_figure(f, settings)
    clean_up_axis(ax, settings)
    clean_up_figure(f, settings)


def plot_timing_vs_nharmonics(settings=default_settings):
    f, ax = plt.subplots(figsize=(settings['axsize'], settings['axsize']))

    settings = translate_color(ax, settings)
    res_fname = lambda n : os.path.join(settings['results_dir'], 'timing_vh_n%d.pkl'%(n))

    hvals = settings['hvals']
    nvals = settings['nvals']

    if not hasattr(nvals, '__iter__'):
        nvals = [ nvals ]

    for n in nvals:

        X, Y, Yerr = generate_random_signal(n, 1.0)

        nh, dt = get_timing_vs_nharmonics(X, Y, Yerr, hvals, filename=res_fname(n),
                         only_use_saved_data=settings['only_use_saved_data'])

        #print "ndata, nharmonics, dlogt / dlogh"
        #for i, h in enumerate(nh):
        #    if i == len(nh) - 1 or i == 0: continue

        #    #avg = lambda arr, j : (1./3.) * float(arr[j+1] + arr[j] + arr[j+1])
        #    avg = lambda arr, j : (1./2.) * float(arr[j+1] + arr[j-1])
        #    diff = lambda arr, j : float(arr[j + 1] - arr[j - 1])

        #    dlogdt = diff(dt, i) / avg(dt, i)
        #    dlogh  = diff(hvals, i) / avg(hvals, i)

        #    print n, h, dlogdt/dlogh

        #print settings['bbox']

        y = np.array(dt)
        nf = int(0.5 * 5 * 5 * n)

        if settings['divide_by'] == "NH":
            y /= nf * np.array(nh)
        elif settings['divide_by'] == "N":
            y /= nf

        #print settings['scatter_params']
        spars = settings['scatter_params']["%d"%(n)]
        zorder = spars['zorder']

        ax.scatter(nh, y, label="$N_{\\rm obs} = %d$\n$N_f=%d$"%(n, nf), **spars)
        lpars = settings['line_params']['%d'%(n)]

        ax.plot(nh, y, zorder = zorder - 1, **lpars)
        #if lpars['ls'] in [ ':', '--', '-.' ]:
        ax.plot(nh, y, zorder = 7, color=ax.get_facecolor(), lw=8)

    # add label for normalization
    divtxt = " / $(H \\times N_f)$" if settings['divide_by'] == "NH"\
       else (" / $N_f$"             if settings['divide_by'] == "N" \
        else "")

    # add axis labels
    ax.set_xlabel("$H$ (number of harmonics)")
    ax.set_ylabel("Exec. time [s]{divtxt}".format(divtxt=divtxt))

    ax.xaxis.set_major_locator(MultipleLocator(5))

    if settings["xlog"]:
        ax.set_xscale('log')
    if settings["ylog"]:
        ax.set_yscale('log')

    ax.set_xlim(*settings['xlim'])


    ylim = None
    if settings['keep_dlogy_equal_to_dlogx']:
        dlogx = log10(settings['xlim'][1]) - log10(settings['xlim'][0])
        ylim  = (settings['ylim'][0], pow(10, log10(settings['ylim'][0]) + dlogx))
    else:
        ylim = settings['ylim']
    ax.set_ylim(*ylim)

    rb = settings['region_boundaries']
    region_boundaries = [ (rb[2*i], rb[2*i+1]) for i in range(len(rb)//2) ]
    region_titles = settings['region_titles']



    for i, (title, boundary) in enumerate(zip(region_titles, region_boundaries)):


        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        b0 = max([ boundary[0], xmin ])
        b1 = min([ boundary[1], xmax ])

        if boundary[0] > xmin and boundary[1] < xmax:
            if i + 1 == settings['highlight_region']:
                ax.fill_between([ boundary[0], boundary[1] ], [ ymin, ymin ], [ymax, ymax],
                                 edgecolor=settings['font_color'], hatch='////', zorder= 2,
                                linewidth=1.0, linestyle=':', lw=0.5, facecolor="none", alpha=1)

                #ax.axvspan(boundary[0], boundary[1], edgecolor=settings['font_color'], ls=':',
                #                facecolor='none', fill=False, lw=1)
            else:
                ax.axvline(boundary[1], ls='-', color=ax.get_facecolor(), lw=3)
                ax.axvline(boundary[1], ls=':', color=settings['font_color'])

        translog    = lambda a : log10(a)
        invtranslog = lambda a : pow(10, a)

        translin    = lambda a : a
        invtranslin = lambda a : a

        transx, invtransx    = (translog, invtranslog) if settings['xlog'] \
                                   else (translin, invtranslin)
        transy, invtransy    = (translog, invtranslog) if settings['ylog'] \
                                   else (translin, invtranslin)

        dx = transx(b1) - transx(b0)
        dxfig = transx(xmax) - transx(xmin)

        xtext = (transx(float(b0)) - transx(xmin) + settings['xtext'] * dx ) / dxfig \
                         + settings['xtext_offset']

        ytext = settings['ytext']
        if dx / dxfig < 0.2:
            xcoord0 = invtransx(transx(b0) + 0.5 * (transx(b1) - transx(b0)))
            ycoord0 = invtransy(transy(b0) + 0.5 * (transy(b1) - transy(b0)))

            coord0 = (xcoord0, ycoord0)

            arrowprops = dict(ec=ax.get_facecolor(), fc=settings['font_color'],
                              lw=1.5, arrowstyle='simple')
            print(settings['bbox'])
            ax.annotate(title, xy=coord0, xycoords='data', xytext=tuple(settings['xyoffset']),
                        textcoords='offset points',
                        horizontalalignment=settings['xtext_ha'], verticalalignment='bottom',
                        arrowprops=arrowprops, color=settings['font_color'],
                        fontsize=settings['annotation_fontsize'], bbox=settings['bbox'])
        else:
            ax.text(xtext, ytext, title, transform=ax.transAxes, ha=settings['xtext_ha'], va='center',
                       fontsize=settings['annotation_fontsize'],
                       color=settings['font_color'], bbox=settings['bbox'])


    ax.legend(loc=settings['legend_loc'], ncol=2, mode='expand')
    #ax.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), mode='expand')

    clean_up_axis(ax, settings)
    adjust_figure(f, settings)
    clean_up_figure(f, settings)

def plot_accuracy(x, y, yerr, y_temp, nharmonics, compare_with=10, settings=default_settings):

    template = Template.from_sampled(y_temp, nharmonics=10)

    # if comparing to large nharmonics, set template now
    if isinstance(compare_with, float) or isinstance(compare_with, int):
        template = Template.from_sampled(y_temp, nharmonics=int(compare_with))
        #template.nharmonics = int(compare_with)
        template.precompute()

    # Set the reference model
    true_model = SlowTemplatePeriodogram(template=template) \
                   if compare_with == 'slow_version' \
                      else FastTemplatePeriodogram(template=template)

    # fit data
    true_model.fit(x, y, yerr)

    results, frq, p_ans = {}, None, None
    label_formatter = lambda kind, h=None : \
                  "$P_{\\rm %s}(\\omega%s)$"\
                          %(kind, "" if h is None else "|H=%d"%(h))
    corrlabel = lambda R : "$R = %.3f$"%(R)

    # store results from the reference model
    # (if the reference model is FastTemplatePeriodogram)
    if isinstance(true_model, FastTemplatePeriodogram):
        frq, p_ans = true_model.autopower()
        results = { 'ans' : (frq, p_ans) }

    # add results from all desired harmonics
    for h in nharmonics:

        # set template harmonics
        #template.nharmonics = h
        template = Template.from_sampled(y_temp, nharmonics=h)
        template.precompute()

        # create & fit modeler
        model = FastTemplatePeriodogram(template=template)
        model.fit(x, y, yerr)

        # compute periodogram
        results[h] = model.autopower()

        # compute results for reference model
        # (if the reference model is gatspy)
        if not 'ans' in results and isinstance(true_model, SlowTemplatePeriodogram):
            frq = results[h][0]
            p_ans = true_model.power(frq)
            results['ans'] = (frq, p_ans)

    # Set up plot geometry
    nplots = len(nharmonics)
    if not settings['nrows'] is None:
        nrows = settings['nrows']
    else:
        nrows = max([ int(sqrt(nplots)), 1 ])

    ncols = 1
    while ncols * nrows < nplots:
        ncols += 1

    figsize = (settings['axsize'] * ncols, settings['axsize'] * nrows)

    # create figure
    f, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # ensure we have a list of axes
    if not hasattr(axes, '__iter__'):
        axes = [ axes ]

    # some plotting definitions
    ans_label = label_formatter('slow') if compare_with == 'slow_version'\
                    else label_formatter('FTP', int(compare_with))

    ax_label_params = dict(fontsize=settings['label_fontsize'],
                              color=settings['font_color'])
    ax_annotation_params = dict(fontsize=settings['annotation_fontsize'],
                                   color=settings['font_color'], bbox=settings['bbox'])
    scatter_params = settings['scatter_params']
    #print scatter_params

    for i, h in enumerate(nharmonics):
        # select the axes instance
        r, c = i / ncols, i % ncols
        ax = axes[c] if ncols >= 1 and nrows == 1 else axes[r][c]

        settings = translate_color(ax, settings)

        p = results[h][1]

        # scatterplot
        ax.plot([0, 1], [0, 1], ls=':', color=settings['font_color'], zorder=2)
        ax.scatter(p, p_ans, **scatter_params)

        # write the pearson R correlation
        ax.text(0.05, 0.9, corrlabel(pearsonr(p_ans, p)[0]), transform=ax.transAxes,
                ha='left', va='bottom', zorder=10, **ax_annotation_params)

        # many of the gatspy periodogram values are 0;
        # write pearson R using only non-zero P_gatspy values
        #if compare_with == 'slow_version':
        #    nonzero_p_ans, nonzero_p = zip(*filter(lambda (Pans, P) : Pans > 0, zip(p_ans, p)))
        #    Rnonzero = pearsonr(nonzero_p_ans, nonzero_p)[0]


        #    ax.text(0.05, 0.9 - 0.03, "%s; $P_{\\rm non-lin. opt.} > 0$"\
        #                  %(corrlabel(Rnonzero)),zorder=10,
        #              transform=ax.transAxes, ha='left', va='top', **ax_annotation_params)

        # set plot properties
        ax.set_xlabel(label_formatter('FTP', h), **ax_label_params)
        if c == 0:
            ax.set_ylabel(ans_label, **ax_label_params)
        else:
            [ label.set_visible(False) for label in ax.get_yticklabels() ]

        ax.set_xlim(settings['x_and_y_min'], 1)
        ax.set_ylim(settings['x_and_y_min'], 1)

        name_list = [ "0", "0.25", "0.5", "0.75", "1" ]
        pos_list  = [ 0, 0.25, 0.5, 0.75, 1 ]
        ax.xaxis.set_major_locator(FixedLocator((pos_list)))
        ax.xaxis.set_major_formatter(FixedFormatter((name_list)))

        ax.yaxis.set_major_locator(FixedLocator((pos_list)))
        ax.yaxis.set_major_formatter(FixedFormatter((name_list)))

        clean_up_axis(ax, settings)

    adjust_figure(f, settings)
    clean_up_figure(f, settings)

def get_multiharmonic_periodogram(x, y, err, nh, hfac=3, ofac=10):
    model = LombScargle(Nterms=nh)
    model.fit(x,y,err)

    pers, p = model.periodogram_auto(nyquist_factor=hfac, oversampling=ofac)

    return np.power(pers, -1), p

def get_bls_periodogram(x, y, err, use_pybls=False, hfac=3, ofac=10, nbin=50, qmin=0.001, qmax=0.5):

    df = 1./(ofac * (max(x) - min(x)))
    Nf = int(floor(0.5 * len(x) * ofac * hfac))
    f0 = 2./(max(x) - min(x))

    if not use_pybls:
        import pyeebls as bls
        #import bls
        u, v = np.empty(len(x)), np.empty(len(x))
        result = bls.eebls(x, y, u, v, Nf, f0, df, nbin, qmin, qmax)
        periodogram, best_period, best_power, depth, q, in1, in2 = result
    else:
        from pybls import BLS
        bls = BLS(x, y, err, fmin=f0, nf=Nf, df=df, nbin=nbin, wmin=qmin, qmax=qmax )
        periodogram  = bls().p


    periodogram -= min(periodogram)
    periodogram /= max(periodogram)

    frq = np.linspace(f0, f0 + Nf * df, Nf)

    return frq, periodogram

def plot_templates_and_periodograms(x, y, err, y_temp, freq_val=None, hfac=None, ofac=None,
                                    nharms = None, settings=default_settings):


    p_ftps = []
    phi_data = None if freq_val is None else (x * freq_val - settings['phi0']) % 1.0
    for i, nharm in enumerate(nharms):

        #model.templates.values()[0].nharmonics = nharm
        #model.templates.values()[0].precompute()
        template = Template.from_sampled(y_temp, nharmonics = nharm)
        template.precompute()
        model = FastTemplatePeriodogram(template=template)

        # Run FTP
        model.fit(x, y, err)

        frq, p = model.autopower(samples_per_peak=ofac, nyquist_factor=hfac)

        p_ftps.append(p)

    # get axes boundaries
    bounds_ax_tmp, bounds_ax_pdg = \
         get_boundaries_for_axtmp_and_axpdg(max(frq), settings)

    # initialize figure
    f = plt.figure(figsize=(2 * settings['axsize'], settings['axsize']))
    ax_pdg = f.add_axes(bounds_ax_pdg)
    ax_tmp = f.add_axes(bounds_ax_tmp)

    settings = translate_color(ax_pdg, settings)

    # get full template
    phi0 = np.linspace(0, 1, len(y_temp))
    y0 = y_temp
    ymin, ymax = min(-y0), max(-y0)

    # x position for text (H = ...); has to be in ax_tmp data coordinates
    x0 = settings['annotate_x0'] / bounds_ax_tmp[2]

    # functions for normalizing templates
    tmpnorm = settings['tmp_height_frac'] / (ymax - ymin)
    yoffset = 0.5 * (1 - settings['tmp_height_frac'])
    tmp_transform = lambda yt, et : ( (-yt - ymin) * tmpnorm + yoffset,
                                                 et * tmpnorm if not et is None else None )

    # normalize data and template
    ydata, edata = tmp_transform(y, err)
    y0, _        = tmp_transform(y0, None)

    #colorfunc = lambda i : "%.5f"%(settings['colorfunc_a'] * (float(len(nharms) - i - 1) / float(len(nharms) - 1)) \
    #                         + settings['colorfunc_b']) if i < len(nharms) - 1 \
    #                          else settings['answer_color']
    colorfunc = lambda i : settings['answer_color']
    for i, (p, h) in enumerate(zip(p_ftps, nharms)):
        offset = len(p_ftps) - i - 1
        ytext =  offset + 0.5
        lw = 1 if i < len(p_ftps) - 1 else 1

        # plot periodogram
        ax_pdg.plot(frq, p + offset, color=colorfunc(i), lw=lw, zorder=20)
        ax_pdg.plot(frq, p + offset, color=ax_pdg.get_facecolor(), lw=3, zorder=19)

        if (i == len(p_ftps) - 1) and settings['plot_bls']:
            frq_bls, p_bls = get_bls_periodogram(x, y, err, hfac=3, ofac=10)
            ax_pdg.plot(frq_bls, p_bls + offset, color=settings['color_bls'],
                            alpha=settings['alpha_bls'], zorder=18)
            ax_pdg.plot(frq_bls, p_bls + offset, color=ax_pdg.get_facecolor(),
                             zorder=17, lw=3)

            ax_pdg.text(0.02, 0.92, "Box Least Squares", color=settings['color_bls'],
                ha='left',
                va='top', bbox=settings['bbox'], fontsize=settings['annotation_fontsize'],
                transform=ax_pdg.transAxes)


        if (i > 0 and settings['plot_multiharmonic_periodogram']):
            # plot multiharmonic periodogram
            frq_mh, p_mh = get_multiharmonic_periodogram(x, y, err, h)

            color = settings['color_multiharmonic_periodogram']
            alpha = settings['alpha_multiharmonic_periodogram']
            ax_pdg.plot(frq_mh, p_mh + offset, color=color, alpha=alpha, zorder=16)
            ax_pdg.plot(frq_mh, p_mh + offset, color=ax_pdg.get_facecolor(), zorder=15, lw=3)

            if i == 1:
                df0 = (2 * freq_val - min(frq_mh)) / (max(frq_mh) - min(frq_mh))
                ind = int(df0 * len(frq_mh))
                dx = max(ax_pdg.get_xlim()) - min(ax_pdg.get_xlim())
                #ax_pdg.annotate('Multiharmonic Lomb Scargle',
                #                xy = (frq_mh[ind], p_mh[ind] + offset + 0.03),
                #                xycoords = 'data',
                #                xytext   = (frq_mh[ind] - 0.015 * dx, 1.45 + offset),
                #                textcoords = 'data',
                #                horizontalalignment = 'right',
                #                verticalalignment   = 'bottom',
                #                color = settings['font_color'],
                #                arrowprops = dict(ec=ax_pdg.get_facecolor(), fc=settings['font_color'],
                #                                    lw=1.5, arrowstyle='simple'),
                #                fontsize  = settings['annotation_fontsize'],
                #                bbox = settings['bbox'])
                ax_pdg.text(0.02, 0.98, "Multi-harmonic Lomb Scargle",
                            fontsize=settings['annotation_fontsize'],
                            bbox    =settings['bbox'],
                            color   = color, ha='left', va='top', transform=ax_pdg.transAxes)




        # get truncated template
        #template.nharmonics = h
        template = Template.from_sampled(y_temp, nharmonics=h)
        template.precompute()

        phi = np.linspace(0, 1, 100)
        ytmp_trunc = template(phi)

        # normalize truncated template
        ytmp_trunc, _ = tmp_transform(ytmp_trunc, None)

        # plot truncated template
        ax_tmp.plot(phi, ytmp_trunc + offset, color=colorfunc(i), lw=lw)

        # plot data
        ax_tmp.errorbar(phi_data, ydata + offset, yerr=edata,
                                            **settings['data_params'])

        # write H = ...
        ax_tmp.text(x0, ytext, "H = %d"%(h), va='center', ha='left',
                            color=settings['font_color'],
                            fontsize=settings['label_fontsize'])


    # Write axis labels
    ax_tmp.set_xlabel('Phase')
    ax_pdg.set_xlabel('Frequency $[d^{-1}]$')

    # Draw line for correct frequency
    if freq_val is not None:
        ax_pdg.axvline(freq_val, ls=':', color='k')

    # Write titles
    ytitle = settings['top'] + settings['title_pad']
    xtitle_pdg = settings['left'] + bounds_ax_tmp[2] + settings['wspace'] \
                                                   + 0.5 * bounds_ax_pdg[2]
    xtitle_tmp = settings['left'] + 0.5 * bounds_ax_tmp[2]

    f.text(xtitle_pdg, ytitle,
        "Template periodogram", va='bottom', ha='center',
        color=settings['font_color'], fontsize=settings['title_fontsize'])

    f.text(xtitle_tmp, ytitle,
        "Template fits", va='bottom', ha='center',
        color=settings['font_color'], fontsize=settings['title_fontsize'])

    # Set other properties
    ax_pdg.set_xlim(0, int(max(frq) / settings['locator_axpdg'])\
                                    * settings['locator_axpdg'])
    ax_pdg.set_ylim(0, len(nharms))

    ax_pdg.xaxis.set_major_locator(MultipleLocator(settings['locator_axpdg']))

    ax_tmp.set_xlim(0, 1)
    ax_tmp.set_ylim(*ax_pdg.get_ylim())
    ax_tmp.set_yticks(ax_pdg.get_yticks())
    ax_tmp.xaxis.set_major_locator(MultipleLocator(settings['locator_axtmp']))

    # for both axes...
    for ax in [ ax_pdg, ax_tmp ]:
        # turn of ytick labels
        [ label.set_visible(False) for label in ax.get_yticklabels()]
        ax.yaxis.set_major_locator(MultipleLocator(1))

        clean_up_axis(ax, settings)

    clean_up_figure(f, settings)

def set_defaults_in_section(settings, defaults):
    #print settings
    for dvar, dval in defaults.iteritems():
        if not dvar in settings:
            settings[dvar] = dval
        elif isinstance(dval, dict):
            settings[dvar] = set_defaults_in_section(settings[dvar], dval)
    return settings

def set_defaults(settings, defaults):
    for group, subsettings in settings.iteritems():
        settings[group] = set_defaults_in_section(subsettings, defaults)
    return settings

def dictprint(d, indent=""):

    for key, value in d.iteritems():
        s = "%s%s-%ds"%(indent, "%", 50 - len(indent))%(key)
        #s = "{fmt}".format(indent=indent, fmt=fmt)
        if isinstance(value, dict):
            print(s)
            dictprint(value, indent = "%s   "%(indent))
        else:
            print("{s}{value}".format(s=s, value=value))

def inject_transit(x, y, q=0.2, freq=100, depth=1):
    phi = (x * freq) % (1.0)
    for i, ph in enumerate(phi):
        if ph > 0.5 * (1 - q) and ph < 0.5 * (1 + q):
            y[i] -= depth

    return y

def test_bls(n = 1000):
    freq = 200
    depth = 1
    q = 0.2

    x, y, err = generate_random_signal(n)

    y = inject_transit(x, y, q=q, freq=freq, depth=depth)

    frq, p = get_bls_periodogram(x, y, err, hfac=3, ofac=10)

    phi = (x * freq)%(1.0)
    f, (ax_lc, ax_bls) = plt.subplots(1, 2, figsize=(10, 5))

    ax_lc.scatter(phi, y, alpha=0.1, marker='o', s=3)
    ax_lc.set_xlim(0, 1)
    ax_lc.set_xlabel('phase')
    ax_lc.set_ylabel('mag')

    ax_bls.plot(frq, p, color='k')
    ax_bls.axvline(freq, ls=':', color='k')
    ax_bls.set_xlabel('freq')
    ax_bls.set_ylabel('bls')

    plt.show()

def plot_nobs_dt_for_surveys(settings=default_settings):
    #surveys = ConfigObj('surveys.ini', unrepr=True)


    surveys_to_use = []
    for survey, info in surveys.iteritems():
        relevant_info = [ 'nobs', 'nlc' ]
        if all([ var in info for var in relevant_info ])\
              and not any([ info[var] is None for var in relevant_info ])\
              and not survey in settings['skip_surveys']:
            surveys_to_use.append(survey)


    f, ax = plt.subplots()
    settings = translate_color(ax, settings)

    conversion = 1./(3600.)
    unit = 'CPU hours'
        #for s in surveys_to_use:
        #    try:
        #        surveys[s]['nobs'] * surveys[s]['nlc']
        #    except:
        #        print s
        #        sys.exit()
    X = [ surveys[s]['nobs'] * surveys[s]['nlc'] for s in surveys_to_use ]
    Yftp = [ conversion * (float(x) / settings['ndata']) * settings['tftp'] for x in X ]
    Ygats = [ conversion * (float(surveys[s]['nobs']) / settings['ndata'])**2 \
                   * surveys[s]['nlc'] * settings['tgats'] for s in surveys_to_use ]

    surveys_to_use, X, Yftp, Ygats = zip(*sorted(zip(surveys_to_use, X, Yftp, Ygats), key=lambda stuff : stuff[1] ))
    ax.scatter(X, Yftp, **settings['scatter_params_ftp'])
    ax.scatter(X, Ygats, **settings['scatter_params_gats'])
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('$N_{\\rm obs} \\times N_{\\rm LC}$')
    ax.set_ylabel("Exec time [%s]"%(unit))

    if 'xlim' in settings:
        ax.set_xlim(*settings['xlim'])
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    y0 = settings['y0']
    yf = settings['yf']
    for i, (survey, x) in enumerate(zip(surveys_to_use, X)):
        common_params = dict(va='center', ha='center', color=settings['font_color'],
                    fontsize=settings['annotation_fontsize'], bbox=settings['bbox'])

        frac = (log10(x) - log10(xmin)) / (log10(xmax) - log10(xmin))
        #ytext = pow(10, ((yf - y0) * frac + y0) * (log10(ymax) - log10(ymin)) + log10(ymin))
        ytext = pow(10, 0.5 * abs(log10(Yftp[i]) - log10(Ygats[i])) \
                                   + min([ log10(Yftp[i]), log10(Ygats[i]) ]))
        #u = 1
        #if i < len(surveys_to_use) - 1 and log10(X[i+1]) - log10(X[i]) < 0.75:
            #ytext = pow(10, 0.6 * (log10(ymax) - log10(ymin)) + log10(ymin))
        ax.plot([x, x], [Yftp[i], Ygats[i]], color=settings['scatter_params_ftp']['color'],
            lw=2)
        if i > 0 and log10(X[i]) - log10(X[i-1]) < settings['dlogx_min']:

            ax.text(x * (1 + settings['dx_text2']), ytext, survey, rotation=270, **common_params)
                    #ha='left', **common_params)
        else:
            ax.text(x * (1 + settings['dx_text']), ytext, survey, rotation='vertical', **common_params)
                    #ha='right', **common_params)
        ax.axvline(x, ls=':', color=settings['font_color'], zorder=1)
    ax.legend(loc='upper left')

    clean_up_axis(ax, settings)
    clean_up_figure(f, settings)

if __name__ == '__main__':

    #test_bls()
    #sys.exit()
    from configobj import ConfigObj

    conf = ConfigObj('plotting.ini', unrepr=True)
    defs = conf['DEFAULT']

    conf = set_defaults(conf, defs)


    template_name = conf['template']['template_name']
    tconf = conf['template'][template_name]
    tconf = set_defaults_in_section(tconf, defs)
    dconf = conf['data']

    template, cn, sn = None, None, None
    x, y, err = None, None, None
    Ttemp, Ytemp = None, None
    ndata, sigma, freq = dconf['ndata'], dconf['sigma'], dconf['freq']
    ofac, hfac = conf['analysis']['ofac'], conf['analysis']['hfac']
    nharms     = conf['analysis']['nharms']


    nh = [ n for n in nharms ]
    nh.append(tconf['nharm_answer'])

    if tconf['fourier']:
        if tconf['template_filename'] is None:
            cn, sn = tconf['cn'], tconf['sn']
            assert(not cn is None and not sn is None)
        else:

            cn, sn = pickle.load(open_results('template_filename', tconf, 'rb'))

        norm = 1./np.sqrt(sum(np.power(cn, 2) + np.power(sn, 2)))
        cn = np.multiply(cn, norm)
        sn = np.multiply(sn, norm)

        nha = tconf['nharm_answer']
        template = Template(c_n=cn[:nha], s_n=sn[:nha])

        y_phase = template(np.linspace(0, 1, 100))

        x, y, err = generate_random_signal(ndata, sigma, freq=freq,
                                              template=template)

    else:
        if tconf['template_filename'] is None:
            # Obtain template from RR Lyrae dataset
            rrl_templates = datasets.rrlyrae.fetch_rrlyrae_templates()

            Ttemp, Ytemp = rrl_templates.get_template(tconf['name'])

        else:
            Ttemp, Ytemp = pickle.load(open_results('template_filename', tconf, 'rb'))

        template = Template.from_sampled(Ytemp, nharmonics=tconf['nharm_answer'])

        x, y, err = generate_random_signal(ndata, sigma, freq=freq, template=template)


    template.precompute()

    # build model
    model = FastTemplatePeriodogram(template=template)
    y_temp = template(np.linspace(0, 1, 100))

    print("plotting timing vs ndata at constant freq")
    plot_timing_vs_ndata_const_freq(settings=conf['timing_vs_ndata_const_freq'])

    print("plotting timing vs nharmonics")
    plot_timing_vs_nharmonics(conf['timing_vs_nharmonics'])

    print("plotting timing vs ndata")
    plot_timing_vs_ndata(conf['timing_vs_ndata'])

    #print("plotting nobs dt for surveys")
    #plot_nobs_dt_for_surveys(settings=conf['nobs_dt_for_surveys'])

    print("plotting templates and periodograms")
    plot_templates_and_periodograms(x, y, err, y_temp, nharms = nh, freq_val=freq, ofac=ofac, hfac=hfac,
                                             settings=conf['templates_and_periodograms'])

    #print("plotting accuracy (self)")
    #plot_accuracy(x, y, err, y_temp, nharms, compare_with=tconf['nharm_answer'],
    #                                         settings=conf['make_accuracy_plot'])

    #print("plotting accuracy (non-linear)")
    #plot_accuracy(x, y, err, y_temp, [ tconf['nharm_answer'] ], compare_with='slow_version',
    #                                         settings=conf['make_accuracy_plot_with_slow_version'])

    plt.show()
