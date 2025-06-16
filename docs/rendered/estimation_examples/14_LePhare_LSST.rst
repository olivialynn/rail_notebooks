RAIL_Lephare example on LSST data
=================================

**Author:** Raphael Shirley, Edited by Tianqing Zhang

**Last successfully run:** May 16, 2025

This example notebook uses synthetic data produced by PZFlow in
combination with several predefined SED templates and filter definition
files.

.. code:: ipython3

    from rail.estimation.algos.lephare import LephareInformer, LephareEstimator
    import numpy as np
    import lephare as lp
    from rail.core.stage import RailStage
    from rail.core.data import TableHandle
    from rail.utils.path_utils import RAILDIR
    
    import matplotlib.pyplot as plt
    import os
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


.. parsed-literal::

    LEPHAREDIR is being set to the default cache directory:
    /home/runner/.cache/lephare/data
    More than 1Gb may be written there.
    LEPHAREWORK is being set to the default cache directory:
    /home/runner/.cache/lephare/work
    Default work cache is already linked. 
    This is linked to the run directory:
    /home/runner/.cache/lephare/runs/20250616T065742


Here we load previously created synthetic data

.. code:: ipython3

    trainFile = os.path.join(RAILDIR, 'rail/examples_data/testdata/output_table_conv_train.hdf5')
    testFile = os.path.join(RAILDIR, 'rail/examples_data/testdata/output_table_conv_test.hdf5')
    traindata_io = DS.read_file("training_data", TableHandle, trainFile)
    testdata_io = DS.read_file("test_data", TableHandle, testFile)

Retrieve all the required filter and template files

One could add or take out bandpasses by editing the configuration file.

.. code:: ipython3

    lephare_config_file = os.path.join(RAILDIR, 'rail/examples_data/estimation_data/data/lsst.para')
    lephare_config = lp.read_config(lephare_config_file)
    
    lp.data_retrieval.get_auxiliary_data(keymap=lephare_config)


.. parsed-literal::

    Number of keywords read in the config file: 86


.. parsed-literal::

    Registry file downloaded and saved as data_registry.txt.


.. parsed-literal::

    Downloading file 'filt/lsst/total_g.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_g.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Downloading file 'filt/lsst/total_u.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_u.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Downloading file 'filt/lsst/total_i.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_i.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Downloading file 'filt/lsst/total_r.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_r.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Downloading file 'filt/lsst/total_y3.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_y3.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Downloading file 'filt/lsst/total_z.pb' from 'https://raw.githubusercontent.com/lephare-photoz/lephare-data/main/filt/lsst/total_z.pb' to '/home/runner/.cache/lephare/data'.


.. parsed-literal::

    Created directory: /home/runner/.cache/lephare/data/filt/lsst
    Checking/downloading 421 files...


.. parsed-literal::

    421 completed.
    All files downloaded successfully and are non-empty.


We use the inform stage to create the library of SEDs with various
redshifts, extinction parameters, and reddening values. This typically
takes ~3-4 minutes.

.. code:: ipython3

    inform_lephare = LephareInformer.make_stage(
        name="inform_Lephare",
        nondetect_val=np.nan,
        model="lephare.pkl",
        hdf5_groupname="",
        # Use a sparse redshift grid to speed up the notebook
        zmin=0,
        zmax=3,
        nzbins=31,
    )
    inform_lephare.inform(traindata_io)


.. parsed-literal::

    rail_lephare is setting the Z_STEP config to 0.1,0.0,3.0 based on the informer params.
    User defined LEPHAREDIR is set. Code runs depend on all required
    auxiliary data being present at /home/runner/.cache/lephare/data.
    User defined LEPHAREWORK is set. All intermediate files will be written to:
     /home/runner/.cache/lephare/runs/inform_Lephare
    # NAME                        IDENT      Lbda_mean Lbeff(Vega)       FWHM     AB-cor    TG-cor      VEGA M_sun(AB)   CALIB      Lb_eff    Fac_corr
    total_u.pb                    1             0.3664      0.3720      0.0458    0.7023   -0.2237  -20.9207    6.3126       0      0.3643      1.0000
    total_g.pb                    2             0.4842      0.4744      0.1395   -0.0845   -0.2511  -20.7220    5.0903       0      0.4763      1.0000
    total_r.pb                    3             0.6251      0.6162      0.1340    0.1519    0.2648  -21.5250    4.6488       0      0.6195      1.0000
    total_i.pb                    4             0.7562      0.7497      0.1297    0.3697    0.5838  -22.1618    4.5362       0      0.7520      1.0000
    total_z.pb                    5             0.8693      0.8670      0.1010    0.5172    0.7617  -22.6166    4.5165       0      0.8670      1.0000
    total_y3.pb                   6             1.0083      1.0053      0.0577    0.5864    0.7956  -23.0094    4.5109       0      1.0065      1.0000
    {'star_sed': '/home/runner/.cache/lephare/data/sed/STAR/STAR_MOD_ALL.list'}


.. parsed-literal::

    #######################################
    # It s translating SEDs to binary library #
    # with the following options :           
    # Config file     : 
    # Library type     : STAR
    # STAR_SED    :/home/runner/.cache/lephare/data/sed/STAR/STAR_MOD_ALL.list
    # STAR_LIB    :LSST_STAR_BIN
    # STAR_LIB doc:/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_STAR_BIN.doc
    # STAR_FSCALE :0.0000
    #######################################
    Number of SED in the list 254


.. parsed-literal::

    WRONG NUMBER OF ARGUMENTS FOR OPTION MOD_EXTINC
    We have 2 instead of 8
    Use default value 0,0 for all filters 


.. parsed-literal::

    #######################################
    # It s computing the SYNTHETIC MAGNITUDES #
    # For Gal/QSO libraries with these OPTIONS #
    # with the following options :           
    # Config file     : 
    # Filter file     : filter_lsst
    # Magnitude type     : AB
    # COSMOLOGY   :70.0000,0.3000,0.7000
    # STAR_LIB_IN    :/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_STAR_BIN(.doc & .bin)
    # STAR_LIB_OUT   :/home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_STAR_MAG(.doc & .bin)
    # LIB_ASCII   YES
    # CREATION_DATE Mon Jun 16 07:22:43 2025
    #############################################
    {'gal_sed': '/home/runner/.cache/lephare/data/sed/GAL/COSMOS_SED/COSMOS_MOD.list'}
    #######################################
    # It s translating SEDs to binary library #
    # with the following options :           
    # Config file     : 
    # Library type     : GAL
    # GAL_SED    :/home/runner/.cache/lephare/data/sed/GAL/COSMOS_SED/COSMOS_MOD.list
    # GAL_LIB    :LSST_GAL_BIN
    # GAL_LIB doc:/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_GAL_BIN.doc
    # GAL_LIB phys:/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_GAL_BIN.phys
    # SEL_AGE    :none
    # GAL_FSCALE :1.0000
    # AGE_RANGE   0.0000 15000000000.0000
    #######################################
    Number of SED in the list 31
    #######################################
    # It s computing the SYNTHETIC MAGNITUDES #
    # For Gal/QSO libraries with these OPTIONS #
    # with the following options :           
    # Config file     : 
    # Filter file     : filter_lsst
    # Magnitude type     : AB
    # GAL_LIB_IN    :/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_GAL_BIN(.doc & .bin)
    # GAL_LIB_OUT   :/home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_GAL_MAG(.doc & .bin)
    # Z_STEP   :0.1000 0.0000 3.0000
    # COSMOLOGY   :70.0000,0.3000,0.7000
    # EXTINC_LAW   :SMC_prevot.dat SB_calzetti.dat SB_calzetti_bump1.dat SB_calzetti_bump2.dat 
    # MOD_EXTINC   :18 26 26 33 26 33 26 33 
    # EB_V   :0.0000 0.0500 0.1000 0.1500 0.2000 0.2500 0.3000 0.3500 0.4000 0.5000 
    # EM_LINES   EMP_UV
    # EM_DISPERSION   0.5000,0.7500,1.0000,1.5000,2.0000,
    # LIB_ASCII   YES
    # CREATION_DATE Mon Jun 16 07:22:43 2025
    #############################################


.. parsed-literal::

    {'qso_sed': '/home/runner/.cache/lephare/data/sed/QSO/SALVATO09/AGN_MOD.list'}
    #######################################
    # It s translating SEDs to binary library #
    # with the following options :           
    # Config file     : 
    # Library type     : QSO
    # QSO_SED    :/home/runner/.cache/lephare/data/sed/QSO/SALVATO09/AGN_MOD.list
    # QSO_LIB    :LSST_QSO_BIN
    # QSO_LIB doc:/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_QSO_BIN.doc
    # QSO_FSCALE :1.0000
    #######################################
    Number of SED in the list 30
    #######################################
    # It s computing the SYNTHETIC MAGNITUDES #
    # For Gal/QSO libraries with these OPTIONS #
    # with the following options :           
    # Config file     : 
    # Filter file     : filter_lsst
    # Magnitude type     : AB
    # QSO_LIB_IN    :/home/runner/.cache/lephare/runs/inform_Lephare/lib_bin/LSST_QSO_BIN(.doc & .bin)
    # QSO_LIB_OUT   :/home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_QSO_MAG(.doc & .bin)


.. parsed-literal::

    # Z_STEP   :0.1000 0.0000 3.0000
    # COSMOLOGY   :70.0000,0.3000,0.7000
    # EXTINC_LAW   :SB_calzetti.dat 
    # MOD_EXTINC   :0 1000 
    # EB_V   :0.0000 0.1000 0.2000 0.3000 # LIB_ASCII   YES
    # CREATION_DATE Mon Jun 16 07:23:06 2025
    #############################################
    Inserting handle into data store.  model_inform_Lephare: inprogress_lephare.pkl, inform_Lephare




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f45330785e0>



Now we take the sythetic test data, and find the best fits from the
library. This results in a PDF, zmode, and zmean for each input test
data. Takes ~2 minutes to run on 1500 inputs.

.. code:: ipython3

    estimate_lephare = LephareEstimator.make_stage(
        name="test_Lephare",
        nondetect_val=np.nan,
        model=inform_lephare.get_handle("model"),
        hdf5_groupname="",
        aliases=dict(input="test_data", output="lephare_estim"),
    )
    
    lephare_estimated = estimate_lephare.estimate(testdata_io)


.. parsed-literal::

    User defined LEPHAREDIR is set. Code runs depend on all required
    auxiliary data being present at /home/runner/.cache/lephare/data.
    User defined LEPHAREWORK is set. All intermediate files will be written to:
     /home/runner/.cache/lephare/runs/inform_Lephare
    Process 0 running estimator on chunk 0 - 1,500
    Using user columns from input table assuming they are in the standard order.
    Processing 1500 objects with 6 bands
    ####################################### 
    # PHOTOMETRIC REDSHIFT with OPTIONS   # 
    # Config file            : 
    # CAT_IN                 : bidon
    # CAT_OUT                : zphot.out
    # CAT_LINES              : 0 1000000000
    # PARA_OUT               : /home/runner/.cache/lephare/data/examples/output.para
    # INP_TYPE               : M
    # CAT_FMT[0:MEME 1:MMEE] : 0
    # CAT_MAG                : AB
    # ZPHOTLIB               : LSST_STAR_MAG LSST_GAL_MAG LSST_QSO_MAG 
    # FIR_LIB                : 
    # FIR_LMIN               : 7.000000
    # FIR_CONT               : -1.000000
    # FIR_SCALE              : -1.000000
    # FIR_FREESCALE          : YES
    # FIR_SUBSTELLAR         : NO
    # ERR_SCALE              : 0.020000 0.020000 0.020000 0.020000 0.020000 0.020000 
    # ERR_FACTOR             : 1.500000 
    # GLB_CONTEXT            : 63
    # FORB_CONTEXT           : -1
    # DZ_WIN                 : 1.000000
    # MIN_THRES              : 0.020000
    # MAG_ABS                : -24.000000 -5.000000
    # MAG_ABS_AGN            : -30.000000 -10.000000
    # MAG_REF                : 3
    # NZ_PRIOR               : -1 -2
    # Z_INTERP               : YES
    # Z_METHOD               : BEST
    # PROB_INTZ              : 0.000000 
    # MABS_METHOD            : 1
    # MABS_CONTEXT           : 63 
    # MABS_REF               : 1 
    # AUTO_ADAPT             : NO
    # ADAPT_BAND             : 5
    # ADAPT_LIM              : 1.500000 23.000000
    # ADAPT_ZBIN             : 0.010000 6.000000
    # ZFIX                   : NO
    # SPEC_OUT               : NO
    # CHI_OUT                : NO
    # PDZ_OUT                : test
    ####################################### 
    Reading input librairies ...
    Read lib 
    Number of keywords to be read in the doc: 13
    Number of keywords read at the command line (excluding -c config): 0
    Reading keywords from /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_QSO_MAG.doc
    Number of keywords read in the config file: 16
    Keyword NUMBER_ROWS not provided 
    Keyword NUMBER_SED not provided 
    Keyword Z_FORM not provided 
    Reading library: /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_QSO_MAG.bin
     Done with the library reading with 3720 SED read. 
    Number of keywords to be read in the doc: 13
    Number of keywords read at the command line (excluding -c config): 0
    Reading keywords from /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_GAL_MAG.doc
    Number of keywords read in the config file: 16
    Keyword NUMBER_ROWS not provided 
    Keyword NUMBER_SED not provided 
    Keyword Z_FORM not provided 
    Reading library: /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_GAL_MAG.bin
     Done with the library reading with 46190 SED read. 
    Number of keywords to be read in the doc: 13
    Number of keywords read at the command line (excluding -c config): 0
    Reading keywords from /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_STAR_MAG.doc
    Number of keywords read in the config file: 16
    Keyword NUMBER_ROWS not provided 
    Keyword NUMBER_SED not provided 
    Keyword Z_FORM not provided 
    Reading library: /home/runner/.cache/lephare/runs/inform_Lephare/lib_mag/LSST_STAR_MAG.bin
     Done with the library reading with 46444 SED read. 
    Read lib out 
    Read filt 
    # NAME                        IDENT      Lbda_mean Lbeff(Vega)       FWHM     AB-cor      VEGA   CALIB    Fac_corr
    total_u.pb                    1             0.3664      0.3720      0.0457    0.7023  -20.9200       0      1.0000
    total_g.pb                    2             0.4842      0.4744      0.1395   -0.0845  -20.7200       0      1.0000
    total_r.pb                    3             0.6251      0.6162      0.1340    0.1519  -21.5300       0      1.0000
    total_i.pb                    4             0.7562      0.7497      0.1297    0.3697  -22.1600       0      1.0000
    total_z.pb                    5             0.8693      0.8670      0.1010    0.5172  -22.6200       0      1.0000
    total_y3.pb                   6             1.0080      1.0050      0.0577    0.5864  -23.0100       0      1.0000


.. parsed-literal::

    AUTO_ADAPT set to NO. Using zero offsets.


.. parsed-literal::

    Source 113 // Band 0 removed to improve the chi2, with old and new chi2 1704.6216 471.2869
    Source 425 // Band 5 removed to improve the chi2, with old and new chi2 1233.4689 0.1301
    Source 449 // Band 0 removed to improve the chi2, with old and new chi2 1778.2322 544.8791
    Source 449 // Band 1 removed to improve the chi2, with old and new chi2 1778.2322 229.4837
    Source 492 // Band 1 removed to improve the chi2, with old and new chi2 3286.5095 2053.2059
    Source 492 // Band 0 removed to improve the chi2, with old and new chi2 3286.5095 819.9700
    Source 789 // Band 5 removed to improve the chi2, with old and new chi2 1121.2501 1.7601
    Source 898 // Band 1 removed to improve the chi2, with old and new chi2 2181.0581 1052.8040
    Source 898 // Band 0 removed to improve the chi2, with old and new chi2 2181.0581 0.1579
    Source 1140 // Band 5 removed to improve the chi2, with old and new chi2 720.3633 1.0712
    Source 1160 // Band 1 removed to improve the chi2, with old and new chi2 2671.7467 1438.7982
    Source 1160 // Band 0 removed to improve the chi2, with old and new chi2 2671.7467 206.0488
    Source 1251 // Band 2 removed to improve the chi2, with old and new chi2 3699.9778 2466.6408
    Source 1251 // Band 1 removed to improve the chi2, with old and new chi2 3699.9778 1233.3111
    Source 1302 // Band 5 removed to improve the chi2, with old and new chi2 1236.5064 3.2175
    Source 1427 // Band 5 removed to improve the chi2, with old and new chi2 861.5093 0.0848
    Source 1451 // Band 5 removed to improve the chi2, with old and new chi2 1174.6267 0.0494
    Inserting handle into data store.  output_test_Lephare: inprogress_output_test_Lephare.hdf5, test_Lephare


An example lephare PDF and comparison to the true value

.. code:: ipython3

    indx = 0
    zgrid = np.linspace(0,3,31)
    plt.plot(zgrid, np.squeeze(lephare_estimated.data[indx].pdf(zgrid)), label='Estimated PDF')
    plt.axvline(x=testdata_io.data['redshift'][indx], color='r', label='True redshift')
    plt.legend()
    plt.xlabel('z')
    plt.show()



.. image:: ../../../docs/rendered/estimation_examples/14_LePhare_LSST_files/../../../docs/rendered/estimation_examples/14_LePhare_LSST_12_0.png


More example fits

.. code:: ipython3

    indxs = [8, 16, 32, 64, 128, 256, 512, 1024]
    zgrid = np.linspace(0,3,31)
    fig, axs = plt.subplots(2,4, figsize=(20,6))
    for i, indx in enumerate(indxs):
        ax = axs[i//4, i%4]
        ax.plot(zgrid, np.squeeze(lephare_estimated.data[indx].pdf(zgrid)), label='Estimated PDF')
        ax.axvline(x=testdata_io.data['redshift'][indx], color='r', label='True redshift')
        ax.set_xlabel('z')



.. image:: ../../../docs/rendered/estimation_examples/14_LePhare_LSST_files/../../../docs/rendered/estimation_examples/14_LePhare_LSST_14_0.png


Histogram of the absolute difference between lephare estimate and true
redshift

.. code:: ipython3

    estimate_diff_from_truth = np.abs(lephare_estimated.data.ancil['zmode'] - testdata_io.data['redshift'])
    
    plt.figure()
    plt.hist(estimate_diff_from_truth, 100)
    plt.xlabel('abs(z_estimated - z_true)')
    plt.show()



.. image:: ../../../docs/rendered/estimation_examples/14_LePhare_LSST_files/../../../docs/rendered/estimation_examples/14_LePhare_LSST_16_0.png


Let’s compare the estimated median redshift vs. the true redshift in a
scatter plot.

.. code:: ipython3

    truez = testdata_io.data['redshift']
    
    plt.figure(figsize=(8,8))
    plt.scatter(truez, lephare_estimated.data.median(), s=3)
    plt.plot([-1,3],[-1,3], 'k--')
    plt.xlim([-0.1,3])
    plt.ylim([-0.1,3])
    
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/14_LePhare_LSST_files/../../../docs/rendered/estimation_examples/14_LePhare_LSST_18_1.png



