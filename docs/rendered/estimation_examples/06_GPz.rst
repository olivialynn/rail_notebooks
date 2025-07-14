GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4032251e-01	 3.1956170e-01	-3.3061352e-01	 3.2499683e-01	[-3.4162823e-01]	 4.6065331e-01


.. parsed-literal::

       2	-2.6555806e-01	 3.0731254e-01	-2.4025165e-01	 3.1378250e-01	[-2.6098588e-01]	 2.3041773e-01


.. parsed-literal::

       3	-2.2052747e-01	 2.8710613e-01	-1.7754511e-01	 2.9557071e-01	[-2.1647979e-01]	 2.8542781e-01


.. parsed-literal::

       4	-1.7921701e-01	 2.6887176e-01	-1.2932246e-01	 2.7976780e-01	[-1.8363201e-01]	 2.8690028e-01


.. parsed-literal::

       5	-1.3150318e-01	 2.5579144e-01	-1.0156043e-01	 2.6404669e-01	[-1.5290067e-01]	 2.0655966e-01
       6	-6.6033927e-02	 2.5143578e-01	-3.7925496e-02	 2.6164816e-01	[-7.7712489e-02]	 1.7777872e-01


.. parsed-literal::

       7	-4.6268429e-02	 2.4732326e-01	-2.2545751e-02	 2.5686628e-01	[-6.0854844e-02]	 2.0150375e-01


.. parsed-literal::

       8	-3.4889223e-02	 2.4540796e-01	-1.4161351e-02	 2.5553656e-01	[-5.6231622e-02]	 2.1232247e-01
       9	-2.0204987e-02	 2.4272770e-01	-2.7175984e-03	 2.5261679e-01	[-4.4082162e-02]	 1.8630719e-01


.. parsed-literal::

      10	-1.0027630e-02	 2.4080152e-01	 5.5183778e-03	 2.5096883e-01	[-3.7732339e-02]	 1.9086099e-01


.. parsed-literal::

      11	-4.3942997e-03	 2.3990778e-01	 9.9950739e-03	 2.4882844e-01	[-2.7262155e-02]	 2.0753646e-01
      12	-6.7071400e-04	 2.3912679e-01	 1.3529163e-02	 2.4827672e-01	[-2.3837041e-02]	 1.9551539e-01


.. parsed-literal::

      13	 3.1824069e-03	 2.3837291e-01	 1.7333590e-02	 2.4783747e-01	[-2.1259128e-02]	 2.0131850e-01


.. parsed-literal::

      14	 5.5563119e-02	 2.2983851e-01	 7.2868983e-02	 2.4051942e-01	[ 3.2843800e-02]	 3.1581450e-01


.. parsed-literal::

      15	 1.0881797e-01	 2.2266793e-01	 1.3049134e-01	 2.3340793e-01	[ 1.0036379e-01]	 3.1245017e-01
      16	 1.6730636e-01	 2.1750186e-01	 1.9086746e-01	 2.2648681e-01	[ 1.5953143e-01]	 1.9933343e-01


.. parsed-literal::

      17	 2.5338221e-01	 2.1629454e-01	 2.8389567e-01	 2.2611549e-01	[ 2.3620748e-01]	 2.0638204e-01


.. parsed-literal::

      18	 2.8933677e-01	 2.1453693e-01	 3.2126235e-01	 2.1986390e-01	[ 2.8751491e-01]	 2.1142054e-01


.. parsed-literal::

      19	 3.4902908e-01	 2.0933257e-01	 3.8224165e-01	 2.1766166e-01	[ 3.5492597e-01]	 2.1238232e-01


.. parsed-literal::

      20	 3.9901370e-01	 2.0705194e-01	 4.3165128e-01	 2.1280338e-01	[ 4.0969136e-01]	 2.0996737e-01


.. parsed-literal::

      21	 4.7685912e-01	 2.0099617e-01	 5.1086073e-01	 2.0862155e-01	[ 4.8255157e-01]	 2.1310759e-01


.. parsed-literal::

      22	 5.7728759e-01	 1.9248136e-01	 6.1570938e-01	 1.9808244e-01	[ 6.0464479e-01]	 2.2201896e-01
      23	 6.3890101e-01	 1.8708560e-01	 6.7807613e-01	 1.9151160e-01	[ 6.6865772e-01]	 1.9687819e-01


.. parsed-literal::

      24	 6.9636631e-01	 1.8225432e-01	 7.3558795e-01	 1.8500700e-01	[ 7.2826644e-01]	 2.0511270e-01


.. parsed-literal::

      25	 7.3352002e-01	 1.7612773e-01	 7.7140694e-01	 1.7638317e-01	[ 7.6513324e-01]	 2.1087885e-01


.. parsed-literal::

      26	 7.6943388e-01	 1.7659202e-01	 8.0813506e-01	 1.7639947e-01	[ 8.0431784e-01]	 2.1885085e-01
      27	 7.9894201e-01	 1.8186848e-01	 8.3833471e-01	 1.8292150e-01	[ 8.2250262e-01]	 1.9261003e-01


.. parsed-literal::

      28	 8.2889355e-01	 1.7702806e-01	 8.6836258e-01	 1.7806192e-01	[ 8.4790134e-01]	 1.8862295e-01


.. parsed-literal::

      29	 8.5779592e-01	 1.7045971e-01	 8.9721115e-01	 1.7195338e-01	[ 8.7628800e-01]	 2.1243715e-01
      30	 8.9923411e-01	 1.6272959e-01	 9.3995928e-01	 1.6460517e-01	[ 9.2872076e-01]	 1.8525624e-01


.. parsed-literal::

      31	 9.1018108e-01	 1.6306640e-01	 9.5143010e-01	 1.6401810e-01	[ 9.4324584e-01]	 2.0620537e-01


.. parsed-literal::

      32	 9.3998929e-01	 1.5982558e-01	 9.8130102e-01	 1.5948927e-01	[ 9.7802628e-01]	 2.1034431e-01


.. parsed-literal::

      33	 9.5599003e-01	 1.5846062e-01	 9.9761642e-01	 1.5716569e-01	[ 9.9516226e-01]	 2.0299602e-01
      34	 9.8050931e-01	 1.5565503e-01	 1.0232166e+00	 1.5282006e-01	[ 1.0209505e+00]	 1.9846630e-01


.. parsed-literal::

      35	 1.0074459e+00	 1.5472797e-01	 1.0515015e+00	 1.5188831e-01	[ 1.0459228e+00]	 2.0817757e-01


.. parsed-literal::

      36	 1.0304372e+00	 1.5229395e-01	 1.0752241e+00	 1.4914543e-01	[ 1.0701154e+00]	 2.1365333e-01
      37	 1.0473700e+00	 1.5072823e-01	 1.0925710e+00	 1.4788449e-01	[ 1.0855301e+00]	 1.9760466e-01


.. parsed-literal::

      38	 1.0570770e+00	 1.4939056e-01	 1.1027390e+00	 1.4781319e-01	  1.0838595e+00 	 2.0118451e-01
      39	 1.0684946e+00	 1.4824082e-01	 1.1139449e+00	 1.4668423e-01	[ 1.0959686e+00]	 1.6977406e-01


.. parsed-literal::

      40	 1.0831448e+00	 1.4659885e-01	 1.1289250e+00	 1.4493657e-01	[ 1.1101529e+00]	 2.1526790e-01
      41	 1.0946566e+00	 1.4497882e-01	 1.1408488e+00	 1.4255735e-01	[ 1.1235399e+00]	 2.0603061e-01


.. parsed-literal::

      42	 1.1033319e+00	 1.4369363e-01	 1.1500396e+00	 1.4130946e-01	[ 1.1251436e+00]	 2.0361757e-01


.. parsed-literal::

      43	 1.1113982e+00	 1.4308340e-01	 1.1579990e+00	 1.4073140e-01	[ 1.1315327e+00]	 2.1221685e-01


.. parsed-literal::

      44	 1.1233012e+00	 1.4191283e-01	 1.1699260e+00	 1.3987446e-01	[ 1.1392459e+00]	 2.2024441e-01
      45	 1.1322496e+00	 1.4108895e-01	 1.1791735e+00	 1.4027926e-01	[ 1.1449780e+00]	 2.0451474e-01


.. parsed-literal::

      46	 1.1394389e+00	 1.3992600e-01	 1.1867597e+00	 1.4007894e-01	[ 1.1580067e+00]	 2.1530485e-01
      47	 1.1488432e+00	 1.3917396e-01	 1.1959572e+00	 1.3988970e-01	[ 1.1677905e+00]	 1.8964124e-01


.. parsed-literal::

      48	 1.1552465e+00	 1.3863014e-01	 1.2023068e+00	 1.3989847e-01	[ 1.1735762e+00]	 2.0593381e-01


.. parsed-literal::

      49	 1.1643127e+00	 1.3789320e-01	 1.2113995e+00	 1.3989032e-01	[ 1.1824651e+00]	 2.1432662e-01
      50	 1.1754052e+00	 1.3688104e-01	 1.2227135e+00	 1.3931217e-01	[ 1.1851512e+00]	 1.9180489e-01


.. parsed-literal::

      51	 1.1873499e+00	 1.3640050e-01	 1.2345842e+00	 1.3930224e-01	[ 1.1976697e+00]	 2.1795940e-01
      52	 1.1959112e+00	 1.3586578e-01	 1.2432365e+00	 1.3849832e-01	[ 1.2040220e+00]	 1.9196796e-01


.. parsed-literal::

      53	 1.2077388e+00	 1.3514773e-01	 1.2556394e+00	 1.3773307e-01	[ 1.2123275e+00]	 2.0521140e-01
      54	 1.2218367e+00	 1.3457702e-01	 1.2704592e+00	 1.3684948e-01	[ 1.2153251e+00]	 1.9520855e-01


.. parsed-literal::

      55	 1.2341639e+00	 1.3422637e-01	 1.2832434e+00	 1.3640929e-01	[ 1.2234080e+00]	 2.0490789e-01
      56	 1.2434709e+00	 1.3389440e-01	 1.2928418e+00	 1.3617588e-01	[ 1.2288580e+00]	 1.8658161e-01


.. parsed-literal::

      57	 1.2541058e+00	 1.3375957e-01	 1.3039560e+00	 1.3651868e-01	[ 1.2346396e+00]	 1.7441559e-01
      58	 1.2661320e+00	 1.3360955e-01	 1.3162295e+00	 1.3651917e-01	[ 1.2347923e+00]	 1.8051934e-01


.. parsed-literal::

      59	 1.2779015e+00	 1.3371740e-01	 1.3281018e+00	 1.3627729e-01	[ 1.2483419e+00]	 1.7679906e-01


.. parsed-literal::

      60	 1.2873866e+00	 1.3351503e-01	 1.3377772e+00	 1.3585794e-01	[ 1.2578579e+00]	 2.1916509e-01


.. parsed-literal::

      61	 1.2963650e+00	 1.3337655e-01	 1.3468407e+00	 1.3509909e-01	[ 1.2680585e+00]	 2.0134521e-01


.. parsed-literal::

      62	 1.3061676e+00	 1.3332766e-01	 1.3565504e+00	 1.3422907e-01	[ 1.2839979e+00]	 2.1277571e-01


.. parsed-literal::

      63	 1.3167368e+00	 1.3330776e-01	 1.3672278e+00	 1.3294879e-01	[ 1.2935341e+00]	 2.0528078e-01
      64	 1.3261388e+00	 1.3313303e-01	 1.3768403e+00	 1.3213684e-01	[ 1.3052205e+00]	 1.7761540e-01


.. parsed-literal::

      65	 1.3336634e+00	 1.3299617e-01	 1.3843880e+00	 1.3154893e-01	[ 1.3105389e+00]	 1.8541050e-01


.. parsed-literal::

      66	 1.3420954e+00	 1.3318401e-01	 1.3928617e+00	 1.3138540e-01	[ 1.3159380e+00]	 2.0586276e-01
      67	 1.3515467e+00	 1.3327957e-01	 1.4027570e+00	 1.3153917e-01	  1.3151124e+00 	 1.9778252e-01


.. parsed-literal::

      68	 1.3602194e+00	 1.3310238e-01	 1.4115548e+00	 1.3203604e-01	[ 1.3190203e+00]	 2.0705462e-01


.. parsed-literal::

      69	 1.3691226e+00	 1.3253001e-01	 1.4204614e+00	 1.3231834e-01	  1.3189677e+00 	 2.0882511e-01
      70	 1.3769474e+00	 1.3169897e-01	 1.4285220e+00	 1.3189734e-01	[ 1.3223047e+00]	 1.9793677e-01


.. parsed-literal::

      71	 1.3839700e+00	 1.3147371e-01	 1.4355608e+00	 1.3178781e-01	[ 1.3257817e+00]	 2.0472717e-01


.. parsed-literal::

      72	 1.3917198e+00	 1.3093372e-01	 1.4433201e+00	 1.3162511e-01	[ 1.3289532e+00]	 2.0516300e-01


.. parsed-literal::

      73	 1.3996943e+00	 1.3059146e-01	 1.4513366e+00	 1.3211293e-01	  1.3274645e+00 	 2.0392632e-01


.. parsed-literal::

      74	 1.4072674e+00	 1.3001449e-01	 1.4589968e+00	 1.3241294e-01	[ 1.3343271e+00]	 2.1562243e-01


.. parsed-literal::

      75	 1.4121434e+00	 1.2950510e-01	 1.4638649e+00	 1.3211486e-01	[ 1.3356864e+00]	 2.0432281e-01
      76	 1.4173337e+00	 1.2921119e-01	 1.4692148e+00	 1.3204674e-01	[ 1.3364131e+00]	 1.8747973e-01


.. parsed-literal::

      77	 1.4233597e+00	 1.2881172e-01	 1.4755347e+00	 1.3194539e-01	  1.3298740e+00 	 2.1542335e-01


.. parsed-literal::

      78	 1.4295259e+00	 1.2874868e-01	 1.4817112e+00	 1.3199851e-01	[ 1.3386811e+00]	 2.1609092e-01


.. parsed-literal::

      79	 1.4337101e+00	 1.2864534e-01	 1.4859259e+00	 1.3195373e-01	[ 1.3411034e+00]	 2.0361018e-01


.. parsed-literal::

      80	 1.4401391e+00	 1.2864797e-01	 1.4926362e+00	 1.3193653e-01	  1.3409046e+00 	 2.1802855e-01
      81	 1.4447870e+00	 1.2808050e-01	 1.4974258e+00	 1.3120785e-01	  1.3406728e+00 	 1.9689941e-01


.. parsed-literal::

      82	 1.4485177e+00	 1.2816213e-01	 1.5011108e+00	 1.3139333e-01	[ 1.3440999e+00]	 1.9811535e-01


.. parsed-literal::

      83	 1.4527132e+00	 1.2831045e-01	 1.5053378e+00	 1.3165090e-01	[ 1.3494229e+00]	 2.1597695e-01
      84	 1.4565710e+00	 1.2832383e-01	 1.5092316e+00	 1.3167919e-01	[ 1.3538174e+00]	 1.9708157e-01


.. parsed-literal::

      85	 1.4627299e+00	 1.2810454e-01	 1.5156824e+00	 1.3131533e-01	[ 1.3594122e+00]	 1.8890047e-01


.. parsed-literal::

      86	 1.4661049e+00	 1.2817278e-01	 1.5190774e+00	 1.3159314e-01	  1.3550661e+00 	 2.1277928e-01


.. parsed-literal::

      87	 1.4696455e+00	 1.2790610e-01	 1.5224330e+00	 1.3125538e-01	  1.3586648e+00 	 2.0399928e-01
      88	 1.4727845e+00	 1.2764424e-01	 1.5256108e+00	 1.3090984e-01	  1.3578139e+00 	 1.8480515e-01


.. parsed-literal::

      89	 1.4781866e+00	 1.2723113e-01	 1.5312050e+00	 1.3056349e-01	  1.3526709e+00 	 1.8185067e-01


.. parsed-literal::

      90	 1.4820783e+00	 1.2696042e-01	 1.5351418e+00	 1.3052303e-01	  1.3485226e+00 	 2.1145582e-01
      91	 1.4865772e+00	 1.2681502e-01	 1.5396374e+00	 1.3069370e-01	  1.3447902e+00 	 1.9825172e-01


.. parsed-literal::

      92	 1.4896458e+00	 1.2670630e-01	 1.5426997e+00	 1.3078707e-01	  1.3478195e+00 	 2.1445704e-01


.. parsed-literal::

      93	 1.4932935e+00	 1.2644906e-01	 1.5464813e+00	 1.3082560e-01	  1.3469065e+00 	 2.0503831e-01
      94	 1.4950838e+00	 1.2603320e-01	 1.5485214e+00	 1.3096612e-01	  1.3438881e+00 	 1.9449687e-01


.. parsed-literal::

      95	 1.4984337e+00	 1.2595978e-01	 1.5517421e+00	 1.3079827e-01	  1.3470801e+00 	 1.9863439e-01


.. parsed-literal::

      96	 1.4998438e+00	 1.2578922e-01	 1.5531381e+00	 1.3060564e-01	  1.3451137e+00 	 2.0984721e-01
      97	 1.5020295e+00	 1.2553566e-01	 1.5553346e+00	 1.3030307e-01	  1.3443563e+00 	 1.9278932e-01


.. parsed-literal::

      98	 1.5057840e+00	 1.2517828e-01	 1.5591641e+00	 1.2968327e-01	  1.3426639e+00 	 2.1606064e-01
      99	 1.5075851e+00	 1.2485911e-01	 1.5612188e+00	 1.2920366e-01	  1.3327554e+00 	 1.8479371e-01


.. parsed-literal::

     100	 1.5111454e+00	 1.2485912e-01	 1.5646649e+00	 1.2894202e-01	  1.3418637e+00 	 2.0673370e-01
     101	 1.5126710e+00	 1.2487787e-01	 1.5661943e+00	 1.2888544e-01	  1.3430617e+00 	 1.7514777e-01


.. parsed-literal::

     102	 1.5149009e+00	 1.2480972e-01	 1.5685000e+00	 1.2854972e-01	  1.3404455e+00 	 1.9128585e-01


.. parsed-literal::

     103	 1.5162550e+00	 1.2469240e-01	 1.5700756e+00	 1.2825525e-01	  1.3372483e+00 	 2.0358419e-01


.. parsed-literal::

     104	 1.5189640e+00	 1.2445429e-01	 1.5726771e+00	 1.2794344e-01	  1.3345572e+00 	 2.1479011e-01


.. parsed-literal::

     105	 1.5206981e+00	 1.2417796e-01	 1.5744085e+00	 1.2756229e-01	  1.3298192e+00 	 2.0214725e-01


.. parsed-literal::

     106	 1.5226702e+00	 1.2386567e-01	 1.5764000e+00	 1.2714674e-01	  1.3238289e+00 	 2.0546985e-01
     107	 1.5265356e+00	 1.2342172e-01	 1.5803095e+00	 1.2658408e-01	  1.3090309e+00 	 1.9978738e-01


.. parsed-literal::

     108	 1.5283451e+00	 1.2319322e-01	 1.5821465e+00	 1.2626830e-01	  1.3034389e+00 	 3.2128024e-01
     109	 1.5309925e+00	 1.2305185e-01	 1.5848172e+00	 1.2610680e-01	  1.2942900e+00 	 1.9297695e-01


.. parsed-literal::

     110	 1.5329637e+00	 1.2309337e-01	 1.5868348e+00	 1.2615059e-01	  1.2926545e+00 	 2.1081996e-01


.. parsed-literal::

     111	 1.5347731e+00	 1.2310046e-01	 1.5887560e+00	 1.2618637e-01	  1.2914130e+00 	 2.0514059e-01
     112	 1.5366645e+00	 1.2308239e-01	 1.5907703e+00	 1.2609271e-01	  1.2815462e+00 	 1.9825935e-01


.. parsed-literal::

     113	 1.5387700e+00	 1.2302874e-01	 1.5929389e+00	 1.2596055e-01	  1.2766555e+00 	 2.0611048e-01


.. parsed-literal::

     114	 1.5410210e+00	 1.2281466e-01	 1.5952266e+00	 1.2560834e-01	  1.2698530e+00 	 2.0206904e-01
     115	 1.5429708e+00	 1.2279398e-01	 1.5971435e+00	 1.2568057e-01	  1.2645124e+00 	 1.6744828e-01


.. parsed-literal::

     116	 1.5445944e+00	 1.2283657e-01	 1.5987281e+00	 1.2584403e-01	  1.2638721e+00 	 1.8579841e-01


.. parsed-literal::

     117	 1.5467343e+00	 1.2298597e-01	 1.6008904e+00	 1.2626058e-01	  1.2574846e+00 	 2.1069837e-01


.. parsed-literal::

     118	 1.5483685e+00	 1.2322824e-01	 1.6026371e+00	 1.2682978e-01	  1.2424670e+00 	 2.1402359e-01
     119	 1.5502573e+00	 1.2338896e-01	 1.6045825e+00	 1.2721035e-01	  1.2366658e+00 	 2.0005655e-01


.. parsed-literal::

     120	 1.5514136e+00	 1.2364308e-01	 1.6059767e+00	 1.2791645e-01	  1.1939360e+00 	 2.0442796e-01
     121	 1.5538103e+00	 1.2348476e-01	 1.6082847e+00	 1.2764820e-01	  1.2047697e+00 	 1.9827676e-01


.. parsed-literal::

     122	 1.5546187e+00	 1.2331235e-01	 1.6090191e+00	 1.2736275e-01	  1.2081323e+00 	 2.1388841e-01


.. parsed-literal::

     123	 1.5570485e+00	 1.2297221e-01	 1.6114000e+00	 1.2693680e-01	  1.1945813e+00 	 2.1340895e-01


.. parsed-literal::

     124	 1.5576588e+00	 1.2265720e-01	 1.6121436e+00	 1.2675352e-01	  1.1929057e+00 	 2.0989347e-01
     125	 1.5598167e+00	 1.2268003e-01	 1.6141975e+00	 1.2687673e-01	  1.1758819e+00 	 1.7153573e-01


.. parsed-literal::

     126	 1.5606416e+00	 1.2274295e-01	 1.6150740e+00	 1.2710672e-01	  1.1671245e+00 	 2.0963359e-01


.. parsed-literal::

     127	 1.5619523e+00	 1.2278444e-01	 1.6164804e+00	 1.2743028e-01	  1.1555672e+00 	 2.0501685e-01


.. parsed-literal::

     128	 1.5639971e+00	 1.2271695e-01	 1.6186761e+00	 1.2775276e-01	  1.1383844e+00 	 2.1666694e-01


.. parsed-literal::

     129	 1.5652496e+00	 1.2260906e-01	 1.6200593e+00	 1.2798234e-01	  1.1166754e+00 	 2.8982329e-01
     130	 1.5671033e+00	 1.2242099e-01	 1.6220106e+00	 1.2807423e-01	  1.1103727e+00 	 1.9731665e-01


.. parsed-literal::

     131	 1.5682494e+00	 1.2230237e-01	 1.6231439e+00	 1.2794459e-01	  1.1122503e+00 	 1.7342687e-01
     132	 1.5697211e+00	 1.2222801e-01	 1.6246010e+00	 1.2792494e-01	  1.1045404e+00 	 1.9369340e-01


.. parsed-literal::

     133	 1.5710820e+00	 1.2217712e-01	 1.6259208e+00	 1.2789835e-01	  1.0890643e+00 	 2.0802879e-01


.. parsed-literal::

     134	 1.5725080e+00	 1.2216175e-01	 1.6273498e+00	 1.2800971e-01	  1.0691945e+00 	 2.0717478e-01


.. parsed-literal::

     135	 1.5743682e+00	 1.2217797e-01	 1.6293225e+00	 1.2833277e-01	  1.0305831e+00 	 2.1739173e-01
     136	 1.5747836e+00	 1.2232512e-01	 1.6298727e+00	 1.2891871e-01	  9.9606189e-01 	 2.0014453e-01


.. parsed-literal::

     137	 1.5760230e+00	 1.2223260e-01	 1.6310323e+00	 1.2866181e-01	  1.0120676e+00 	 2.1667743e-01


.. parsed-literal::

     138	 1.5768158e+00	 1.2218614e-01	 1.6318424e+00	 1.2865685e-01	  1.0137728e+00 	 2.0462155e-01


.. parsed-literal::

     139	 1.5777270e+00	 1.2214893e-01	 1.6327859e+00	 1.2870087e-01	  1.0080904e+00 	 2.0381784e-01


.. parsed-literal::

     140	 1.5790370e+00	 1.2215855e-01	 1.6340974e+00	 1.2895316e-01	  9.9804404e-01 	 2.0969105e-01


.. parsed-literal::

     141	 1.5798685e+00	 1.2212713e-01	 1.6349522e+00	 1.2914679e-01	  9.7559714e-01 	 3.1186557e-01


.. parsed-literal::

     142	 1.5808941e+00	 1.2214938e-01	 1.6359151e+00	 1.2940073e-01	  9.6642219e-01 	 2.1712613e-01


.. parsed-literal::

     143	 1.5816956e+00	 1.2213349e-01	 1.6366641e+00	 1.2955357e-01	  9.5773088e-01 	 2.0868492e-01


.. parsed-literal::

     144	 1.5827577e+00	 1.2211087e-01	 1.6377047e+00	 1.2976820e-01	  9.4290926e-01 	 2.0810986e-01
     145	 1.5834604e+00	 1.2205147e-01	 1.6385198e+00	 1.3017216e-01	  9.3499838e-01 	 1.9992542e-01


.. parsed-literal::

     146	 1.5848118e+00	 1.2205296e-01	 1.6398340e+00	 1.3018798e-01	  9.1982858e-01 	 2.0928741e-01


.. parsed-literal::

     147	 1.5854650e+00	 1.2205327e-01	 1.6405220e+00	 1.3017528e-01	  9.1898193e-01 	 2.1170187e-01
     148	 1.5865557e+00	 1.2205139e-01	 1.6417196e+00	 1.3023455e-01	  9.1731749e-01 	 1.8656468e-01


.. parsed-literal::

     149	 1.5879824e+00	 1.2199885e-01	 1.6433062e+00	 1.3026223e-01	  9.2097903e-01 	 2.1112013e-01


.. parsed-literal::

     150	 1.5895517e+00	 1.2194230e-01	 1.6450263e+00	 1.3029411e-01	  9.1030444e-01 	 2.0720196e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4aac3cadd0>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.1 s, sys: 45.9 ms, total: 2.14 s
    Wall time: 648 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

