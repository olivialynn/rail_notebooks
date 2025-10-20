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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4862494e-01	 3.2167079e-01	-3.3889999e-01	 3.1640505e-01	[-3.2865058e-01]	 4.6318936e-01


.. parsed-literal::

       2	-2.7465647e-01	 3.1030935e-01	-2.5000154e-01	 3.0523607e-01	[-2.3555320e-01]	 2.3269820e-01


.. parsed-literal::

       3	-2.2901755e-01	 2.8941064e-01	-1.8615355e-01	 2.8767500e-01	[-1.7975867e-01]	 2.7806759e-01


.. parsed-literal::

       4	-1.8873278e-01	 2.7136452e-01	-1.3907391e-01	 2.7445592e-01	[-1.5043258e-01]	 3.0476475e-01


.. parsed-literal::

       5	-1.3458404e-01	 2.5873671e-01	-1.0352826e-01	 2.5848258e-01	[-1.0170817e-01]	 2.0646024e-01


.. parsed-literal::

       6	-7.7361447e-02	 2.5415761e-01	-4.8804265e-02	 2.5239187e-01	[-4.3076134e-02]	 2.0958638e-01


.. parsed-literal::

       7	-5.7673694e-02	 2.5007613e-01	-3.3500879e-02	 2.4785934e-01	[-2.5640732e-02]	 2.1141601e-01


.. parsed-literal::

       8	-4.5560396e-02	 2.4806291e-01	-2.4734824e-02	 2.4530436e-01	[-1.4266136e-02]	 2.0177197e-01


.. parsed-literal::

       9	-2.7668089e-02	 2.4460461e-01	-1.0449104e-02	 2.4297935e-01	[-3.1926221e-03]	 2.1657443e-01
      10	-1.6452494e-02	 2.4247224e-01	-1.1564778e-03	 2.4255071e-01	[ 2.7532573e-04]	 2.0045805e-01


.. parsed-literal::

      11	-1.0681040e-02	 2.4135699e-01	 3.9287705e-03	 2.4254555e-01	[ 1.0118514e-03]	 2.0236444e-01


.. parsed-literal::

      12	-5.9064867e-03	 2.4053228e-01	 8.1525212e-03	 2.4283034e-01	  4.7188700e-04 	 2.0726967e-01
      13	 4.2881025e-05	 2.3935413e-01	 1.4242961e-02	 2.4181934e-01	[ 4.9051219e-03]	 2.0171475e-01


.. parsed-literal::

      14	 2.3007551e-02	 2.3547943e-01	 3.8363877e-02	 2.3708646e-01	[ 2.9173057e-02]	 1.9964504e-01


.. parsed-literal::

      15	 9.5674317e-02	 2.2355087e-01	 1.1704562e-01	 2.2636542e-01	[ 1.0374782e-01]	 3.1289673e-01
      16	 1.5239306e-01	 2.1895996e-01	 1.7422954e-01	 2.2444792e-01	[ 1.5277634e-01]	 1.8840671e-01


.. parsed-literal::

      17	 2.5715560e-01	 2.1399249e-01	 2.8487748e-01	 2.2273335e-01	[ 2.3709829e-01]	 1.9960833e-01


.. parsed-literal::

      18	 3.0339910e-01	 2.1140380e-01	 3.3355872e-01	 2.2134096e-01	[ 2.7733209e-01]	 2.1173429e-01


.. parsed-literal::

      19	 3.6864175e-01	 2.0813103e-01	 4.0194615e-01	 2.1418940e-01	[ 3.5024921e-01]	 2.1746039e-01


.. parsed-literal::

      20	 4.3017938e-01	 2.0796804e-01	 4.6395210e-01	 2.0982928e-01	[ 4.0976157e-01]	 2.0914626e-01


.. parsed-literal::

      21	 4.9412365e-01	 2.0248877e-01	 5.2883640e-01	 2.0476162e-01	[ 4.5870269e-01]	 2.1732974e-01
      22	 5.7015514e-01	 1.9800963e-01	 6.0663356e-01	 2.0088742e-01	[ 5.1338551e-01]	 1.7262816e-01


.. parsed-literal::

      23	 6.2071442e-01	 1.9359179e-01	 6.6059460e-01	 1.9859402e-01	[ 5.3025548e-01]	 1.7926764e-01
      24	 6.6472750e-01	 1.9267527e-01	 7.0427834e-01	 1.9548951e-01	[ 5.9239830e-01]	 1.8651009e-01


.. parsed-literal::

      25	 6.9956214e-01	 1.8955711e-01	 7.3840819e-01	 1.9197684e-01	[ 6.3037119e-01]	 2.1357751e-01


.. parsed-literal::

      26	 7.2951536e-01	 1.8999366e-01	 7.6799159e-01	 1.9172186e-01	[ 6.5860995e-01]	 2.1783924e-01


.. parsed-literal::

      27	 7.5940839e-01	 1.9016896e-01	 7.9720179e-01	 1.9455035e-01	[ 6.8912491e-01]	 2.0777774e-01
      28	 7.7495221e-01	 1.9585515e-01	 8.1288746e-01	 2.0033992e-01	[ 7.0194980e-01]	 1.9889140e-01


.. parsed-literal::

      29	 7.8832654e-01	 1.8998180e-01	 8.2742716e-01	 1.9726224e-01	[ 7.0386916e-01]	 2.0319223e-01


.. parsed-literal::

      30	 8.0347244e-01	 1.8928360e-01	 8.4214641e-01	 1.9791317e-01	[ 7.1900568e-01]	 2.0324922e-01


.. parsed-literal::

      31	 8.2137347e-01	 1.8921235e-01	 8.6029094e-01	 1.9985748e-01	[ 7.3055355e-01]	 2.0783782e-01


.. parsed-literal::

      32	 8.4098199e-01	 1.8532088e-01	 8.7999380e-01	 1.9571668e-01	[ 7.5557431e-01]	 2.0920610e-01


.. parsed-literal::

      33	 8.6574496e-01	 1.8279122e-01	 9.0540441e-01	 1.9187501e-01	[ 7.8565676e-01]	 2.0965767e-01


.. parsed-literal::

      34	 8.9366094e-01	 1.8184244e-01	 9.3421576e-01	 1.8721794e-01	[ 8.2556039e-01]	 2.1001792e-01
      35	 9.1046355e-01	 1.8257539e-01	 9.5101290e-01	 1.8541117e-01	[ 8.5758726e-01]	 1.9318581e-01


.. parsed-literal::

      36	 9.2150935e-01	 1.8145893e-01	 9.6221081e-01	 1.8415005e-01	[ 8.6557542e-01]	 2.0930505e-01


.. parsed-literal::

      37	 9.3512108e-01	 1.8155288e-01	 9.7621079e-01	 1.8331320e-01	[ 8.7683492e-01]	 2.0744562e-01


.. parsed-literal::

      38	 9.4870122e-01	 1.7959817e-01	 9.9084165e-01	 1.8034519e-01	[ 8.8889242e-01]	 2.0644712e-01


.. parsed-literal::

      39	 9.6900263e-01	 1.7896181e-01	 1.0115124e+00	 1.7715805e-01	[ 9.1568881e-01]	 2.1524549e-01


.. parsed-literal::

      40	 9.8799535e-01	 1.7876437e-01	 1.0308498e+00	 1.7509798e-01	[ 9.4241226e-01]	 2.0658612e-01


.. parsed-literal::

      41	 1.0120298e+00	 1.7600683e-01	 1.0559557e+00	 1.7197790e-01	[ 9.6481598e-01]	 2.0886183e-01


.. parsed-literal::

      42	 1.0344490e+00	 1.7423632e-01	 1.0790125e+00	 1.6929165e-01	[ 9.9665420e-01]	 2.0985150e-01


.. parsed-literal::

      43	 1.0465868e+00	 1.7111263e-01	 1.0916957e+00	 1.6833208e-01	[ 1.0054493e+00]	 2.1024585e-01


.. parsed-literal::

      44	 1.0567451e+00	 1.6949769e-01	 1.1015511e+00	 1.6715334e-01	[ 1.0155469e+00]	 2.1095943e-01


.. parsed-literal::

      45	 1.0678944e+00	 1.6691622e-01	 1.1127206e+00	 1.6617358e-01	[ 1.0248517e+00]	 2.0362306e-01
      46	 1.0825255e+00	 1.6308926e-01	 1.1278480e+00	 1.6386038e-01	[ 1.0358507e+00]	 1.9525647e-01


.. parsed-literal::

      47	 1.0974402e+00	 1.5603241e-01	 1.1435656e+00	 1.5997899e-01	[ 1.0413163e+00]	 1.7149949e-01
      48	 1.1110380e+00	 1.5558404e-01	 1.1567876e+00	 1.5841979e-01	[ 1.0539244e+00]	 1.9560194e-01


.. parsed-literal::

      49	 1.1189723e+00	 1.5553511e-01	 1.1644485e+00	 1.5784161e-01	[ 1.0604297e+00]	 2.0256495e-01


.. parsed-literal::

      50	 1.1322653e+00	 1.5285713e-01	 1.1779810e+00	 1.5558520e-01	[ 1.0678361e+00]	 2.0790529e-01
      51	 1.1445164e+00	 1.4788564e-01	 1.1909013e+00	 1.5189008e-01	[ 1.0731077e+00]	 1.8193221e-01


.. parsed-literal::

      52	 1.1566415e+00	 1.4639378e-01	 1.2030443e+00	 1.5113351e-01	[ 1.0885786e+00]	 2.1205473e-01
      53	 1.1661033e+00	 1.4528076e-01	 1.2123844e+00	 1.5054429e-01	[ 1.1016979e+00]	 1.7918015e-01


.. parsed-literal::

      54	 1.1830762e+00	 1.4287765e-01	 1.2298451e+00	 1.4924208e-01	[ 1.1176984e+00]	 1.8139005e-01


.. parsed-literal::

      55	 1.1913435e+00	 1.4133591e-01	 1.2386275e+00	 1.4785689e-01	[ 1.1275440e+00]	 3.0096722e-01


.. parsed-literal::

      56	 1.2004371e+00	 1.4049153e-01	 1.2480715e+00	 1.4651012e-01	[ 1.1352773e+00]	 2.1332455e-01


.. parsed-literal::

      57	 1.2104009e+00	 1.3878585e-01	 1.2585155e+00	 1.4458368e-01	[ 1.1375875e+00]	 2.0980620e-01


.. parsed-literal::

      58	 1.2202027e+00	 1.3743691e-01	 1.2686041e+00	 1.4287534e-01	[ 1.1417348e+00]	 2.1033049e-01
      59	 1.2298910e+00	 1.3678600e-01	 1.2785202e+00	 1.4135879e-01	[ 1.1454086e+00]	 1.7419028e-01


.. parsed-literal::

      60	 1.2375500e+00	 1.3655162e-01	 1.2861697e+00	 1.4054819e-01	[ 1.1546829e+00]	 1.7924523e-01
      61	 1.2476870e+00	 1.3625734e-01	 1.2965023e+00	 1.3973134e-01	[ 1.1686207e+00]	 1.9602585e-01


.. parsed-literal::

      62	 1.2545624e+00	 1.3602914e-01	 1.3036102e+00	 1.3904797e-01	[ 1.1767374e+00]	 2.1865296e-01


.. parsed-literal::

      63	 1.2630743e+00	 1.3549325e-01	 1.3119937e+00	 1.3897916e-01	[ 1.1877535e+00]	 2.0275187e-01
      64	 1.2703007e+00	 1.3480639e-01	 1.3193532e+00	 1.3868661e-01	[ 1.1959092e+00]	 1.7288733e-01


.. parsed-literal::

      65	 1.2772166e+00	 1.3415388e-01	 1.3264626e+00	 1.3827785e-01	[ 1.2017323e+00]	 2.0568275e-01


.. parsed-literal::

      66	 1.2857588e+00	 1.3324461e-01	 1.3354325e+00	 1.3782688e-01	[ 1.2139655e+00]	 2.0490098e-01


.. parsed-literal::

      67	 1.2934357e+00	 1.3312430e-01	 1.3430947e+00	 1.3735784e-01	[ 1.2203723e+00]	 2.0961475e-01
      68	 1.2985726e+00	 1.3291752e-01	 1.3480558e+00	 1.3734033e-01	[ 1.2240345e+00]	 1.9866109e-01


.. parsed-literal::

      69	 1.3084462e+00	 1.3299779e-01	 1.3579631e+00	 1.3770121e-01	[ 1.2323056e+00]	 1.9648027e-01
      70	 1.3123327e+00	 1.3272040e-01	 1.3624501e+00	 1.3774925e-01	  1.2182775e+00 	 1.9860625e-01


.. parsed-literal::

      71	 1.3199632e+00	 1.3258693e-01	 1.3697702e+00	 1.3746882e-01	[ 1.2326012e+00]	 1.9667077e-01


.. parsed-literal::

      72	 1.3240298e+00	 1.3247119e-01	 1.3738424e+00	 1.3734988e-01	[ 1.2378855e+00]	 2.0565224e-01


.. parsed-literal::

      73	 1.3289872e+00	 1.3221289e-01	 1.3788926e+00	 1.3710150e-01	[ 1.2419630e+00]	 2.1078205e-01
      74	 1.3371465e+00	 1.3158733e-01	 1.3871916e+00	 1.3674823e-01	[ 1.2470841e+00]	 1.9803190e-01


.. parsed-literal::

      75	 1.3404061e+00	 1.3211088e-01	 1.3908114e+00	 1.3568939e-01	  1.2443391e+00 	 2.0825791e-01


.. parsed-literal::

      76	 1.3502038e+00	 1.3127884e-01	 1.4003123e+00	 1.3535399e-01	[ 1.2521920e+00]	 2.0608211e-01


.. parsed-literal::

      77	 1.3544373e+00	 1.3095163e-01	 1.4045803e+00	 1.3518467e-01	[ 1.2524197e+00]	 2.0653820e-01


.. parsed-literal::

      78	 1.3594920e+00	 1.3067451e-01	 1.4097800e+00	 1.3459865e-01	  1.2521323e+00 	 2.0766139e-01


.. parsed-literal::

      79	 1.3638460e+00	 1.3109193e-01	 1.4145664e+00	 1.3390510e-01	  1.2505198e+00 	 2.0859432e-01


.. parsed-literal::

      80	 1.3697692e+00	 1.3053322e-01	 1.4204846e+00	 1.3348904e-01	[ 1.2568969e+00]	 2.1288967e-01


.. parsed-literal::

      81	 1.3742523e+00	 1.3028198e-01	 1.4249161e+00	 1.3300032e-01	[ 1.2628815e+00]	 2.0707560e-01
      82	 1.3803856e+00	 1.2972093e-01	 1.4311596e+00	 1.3169493e-01	[ 1.2674754e+00]	 1.8582678e-01


.. parsed-literal::

      83	 1.3845047e+00	 1.2982953e-01	 1.4354283e+00	 1.2999647e-01	[ 1.2772506e+00]	 2.1430111e-01
      84	 1.3912289e+00	 1.2921501e-01	 1.4421867e+00	 1.2904437e-01	[ 1.2794842e+00]	 1.8832040e-01


.. parsed-literal::

      85	 1.3948907e+00	 1.2883480e-01	 1.4458573e+00	 1.2866965e-01	[ 1.2797174e+00]	 2.1017623e-01


.. parsed-literal::

      86	 1.4005370e+00	 1.2835742e-01	 1.4516429e+00	 1.2794007e-01	[ 1.2810020e+00]	 2.1061921e-01


.. parsed-literal::

      87	 1.4055330e+00	 1.2806588e-01	 1.4568508e+00	 1.2677342e-01	  1.2756955e+00 	 2.1776867e-01
      88	 1.4112427e+00	 1.2806471e-01	 1.4626201e+00	 1.2627416e-01	[ 1.2824826e+00]	 1.9907594e-01


.. parsed-literal::

      89	 1.4146930e+00	 1.2817054e-01	 1.4659818e+00	 1.2627940e-01	[ 1.2880222e+00]	 1.8069792e-01
      90	 1.4190479e+00	 1.2829211e-01	 1.4702957e+00	 1.2625958e-01	[ 1.2924410e+00]	 1.7431068e-01


.. parsed-literal::

      91	 1.4240802e+00	 1.2846891e-01	 1.4753310e+00	 1.2580786e-01	[ 1.2973509e+00]	 2.1310234e-01
      92	 1.4287361e+00	 1.2824150e-01	 1.4800819e+00	 1.2575490e-01	[ 1.2998772e+00]	 1.9921994e-01


.. parsed-literal::

      93	 1.4320858e+00	 1.2801662e-01	 1.4834018e+00	 1.2544712e-01	[ 1.3003189e+00]	 2.1466064e-01


.. parsed-literal::

      94	 1.4353341e+00	 1.2766569e-01	 1.4867375e+00	 1.2507070e-01	  1.2979099e+00 	 2.0428634e-01


.. parsed-literal::

      95	 1.4392110e+00	 1.2748860e-01	 1.4907583e+00	 1.2496745e-01	  1.2927512e+00 	 2.1292138e-01


.. parsed-literal::

      96	 1.4433030e+00	 1.2737471e-01	 1.4952990e+00	 1.2509820e-01	  1.2761139e+00 	 2.1850896e-01


.. parsed-literal::

      97	 1.4484909e+00	 1.2741147e-01	 1.5003256e+00	 1.2511022e-01	  1.2818265e+00 	 2.1003366e-01
      98	 1.4505952e+00	 1.2744571e-01	 1.5023302e+00	 1.2517229e-01	  1.2863499e+00 	 1.8489289e-01


.. parsed-literal::

      99	 1.4553478e+00	 1.2736026e-01	 1.5071134e+00	 1.2521572e-01	  1.2900413e+00 	 1.9912505e-01


.. parsed-literal::

     100	 1.4601233e+00	 1.2725453e-01	 1.5120806e+00	 1.2563536e-01	  1.2818471e+00 	 2.1041632e-01
     101	 1.4644859e+00	 1.2689955e-01	 1.5165357e+00	 1.2541753e-01	  1.2866583e+00 	 1.9905758e-01


.. parsed-literal::

     102	 1.4673798e+00	 1.2665886e-01	 1.5194278e+00	 1.2513822e-01	  1.2860795e+00 	 1.8490696e-01
     103	 1.4708326e+00	 1.2620746e-01	 1.5230460e+00	 1.2507746e-01	  1.2790637e+00 	 1.9596291e-01


.. parsed-literal::

     104	 1.4742418e+00	 1.2602055e-01	 1.5265679e+00	 1.2453763e-01	  1.2776804e+00 	 1.8309617e-01


.. parsed-literal::

     105	 1.4781547e+00	 1.2569876e-01	 1.5305383e+00	 1.2450074e-01	  1.2756740e+00 	 2.0253825e-01
     106	 1.4811911e+00	 1.2543077e-01	 1.5335777e+00	 1.2451544e-01	  1.2764508e+00 	 1.7209792e-01


.. parsed-literal::

     107	 1.4832049e+00	 1.2540342e-01	 1.5355118e+00	 1.2447304e-01	  1.2794427e+00 	 2.0816422e-01


.. parsed-literal::

     108	 1.4860874e+00	 1.2529838e-01	 1.5383707e+00	 1.2444070e-01	  1.2799921e+00 	 2.1172047e-01


.. parsed-literal::

     109	 1.4883778e+00	 1.2521335e-01	 1.5408544e+00	 1.2431268e-01	  1.2767161e+00 	 2.0438576e-01


.. parsed-literal::

     110	 1.4916467e+00	 1.2504117e-01	 1.5441228e+00	 1.2424345e-01	  1.2765469e+00 	 2.0227146e-01
     111	 1.4932522e+00	 1.2495183e-01	 1.5457728e+00	 1.2419807e-01	  1.2750970e+00 	 1.9582939e-01


.. parsed-literal::

     112	 1.4960407e+00	 1.2474223e-01	 1.5486725e+00	 1.2410739e-01	  1.2734752e+00 	 2.0286632e-01
     113	 1.4992468e+00	 1.2453757e-01	 1.5519335e+00	 1.2406663e-01	  1.2726807e+00 	 1.8082762e-01


.. parsed-literal::

     114	 1.5013755e+00	 1.2421693e-01	 1.5541261e+00	 1.2438935e-01	  1.2774887e+00 	 1.9925237e-01


.. parsed-literal::

     115	 1.5040436e+00	 1.2427537e-01	 1.5566351e+00	 1.2439703e-01	  1.2800109e+00 	 2.0530725e-01


.. parsed-literal::

     116	 1.5054543e+00	 1.2425112e-01	 1.5579663e+00	 1.2448832e-01	  1.2818836e+00 	 2.0489669e-01
     117	 1.5079440e+00	 1.2414496e-01	 1.5604682e+00	 1.2448134e-01	  1.2817030e+00 	 2.0011997e-01


.. parsed-literal::

     118	 1.5093649e+00	 1.2385391e-01	 1.5620955e+00	 1.2501441e-01	  1.2731273e+00 	 2.0580983e-01
     119	 1.5124647e+00	 1.2373819e-01	 1.5651393e+00	 1.2459195e-01	  1.2774210e+00 	 1.8566585e-01


.. parsed-literal::

     120	 1.5138337e+00	 1.2362727e-01	 1.5665506e+00	 1.2437822e-01	  1.2762319e+00 	 2.1982026e-01
     121	 1.5157877e+00	 1.2342483e-01	 1.5685628e+00	 1.2406510e-01	  1.2756544e+00 	 1.9259858e-01


.. parsed-literal::

     122	 1.5181314e+00	 1.2327358e-01	 1.5709938e+00	 1.2366089e-01	  1.2675216e+00 	 2.0813656e-01
     123	 1.5209136e+00	 1.2309098e-01	 1.5737300e+00	 1.2345601e-01	  1.2715849e+00 	 1.7111707e-01


.. parsed-literal::

     124	 1.5230408e+00	 1.2299473e-01	 1.5757962e+00	 1.2332847e-01	  1.2752936e+00 	 2.0330548e-01
     125	 1.5249195e+00	 1.2292619e-01	 1.5776603e+00	 1.2318396e-01	  1.2770017e+00 	 1.9116759e-01


.. parsed-literal::

     126	 1.5262770e+00	 1.2284012e-01	 1.5790566e+00	 1.2302278e-01	  1.2792590e+00 	 2.8909993e-01


.. parsed-literal::

     127	 1.5278997e+00	 1.2278554e-01	 1.5807026e+00	 1.2287733e-01	  1.2784308e+00 	 2.0964003e-01


.. parsed-literal::

     128	 1.5295387e+00	 1.2268745e-01	 1.5824327e+00	 1.2270189e-01	  1.2762232e+00 	 2.0815945e-01


.. parsed-literal::

     129	 1.5316753e+00	 1.2256785e-01	 1.5846784e+00	 1.2250994e-01	  1.2758992e+00 	 2.0855498e-01


.. parsed-literal::

     130	 1.5345510e+00	 1.2239280e-01	 1.5877767e+00	 1.2212725e-01	  1.2723302e+00 	 2.0854092e-01


.. parsed-literal::

     131	 1.5363876e+00	 1.2232810e-01	 1.5897592e+00	 1.2219081e-01	  1.2830114e+00 	 2.0559025e-01
     132	 1.5383801e+00	 1.2226205e-01	 1.5916311e+00	 1.2220337e-01	  1.2822417e+00 	 1.9301534e-01


.. parsed-literal::

     133	 1.5394639e+00	 1.2220922e-01	 1.5926740e+00	 1.2231862e-01	  1.2824341e+00 	 2.0318699e-01
     134	 1.5411203e+00	 1.2209412e-01	 1.5943454e+00	 1.2249649e-01	  1.2843010e+00 	 1.8430638e-01


.. parsed-literal::

     135	 1.5427081e+00	 1.2204751e-01	 1.5960104e+00	 1.2306479e-01	  1.2779011e+00 	 1.8737483e-01


.. parsed-literal::

     136	 1.5446046e+00	 1.2191627e-01	 1.5979159e+00	 1.2313399e-01	  1.2819531e+00 	 2.1557736e-01
     137	 1.5459389e+00	 1.2181615e-01	 1.5993057e+00	 1.2319443e-01	  1.2827062e+00 	 1.8622828e-01


.. parsed-literal::

     138	 1.5472628e+00	 1.2172681e-01	 1.6006621e+00	 1.2331600e-01	  1.2807012e+00 	 1.8492460e-01
     139	 1.5486715e+00	 1.2147808e-01	 1.6022391e+00	 1.2363641e-01	  1.2733789e+00 	 1.9943523e-01


.. parsed-literal::

     140	 1.5501168e+00	 1.2148328e-01	 1.6036080e+00	 1.2367356e-01	  1.2712490e+00 	 2.0519304e-01


.. parsed-literal::

     141	 1.5508418e+00	 1.2145393e-01	 1.6043038e+00	 1.2368323e-01	  1.2706302e+00 	 2.0864224e-01
     142	 1.5524740e+00	 1.2134301e-01	 1.6059333e+00	 1.2361595e-01	  1.2692209e+00 	 1.9935417e-01


.. parsed-literal::

     143	 1.5544911e+00	 1.2114666e-01	 1.6080962e+00	 1.2351316e-01	  1.2688733e+00 	 2.0952129e-01
     144	 1.5562621e+00	 1.2104055e-01	 1.6099098e+00	 1.2317543e-01	  1.2649476e+00 	 1.7981458e-01


.. parsed-literal::

     145	 1.5573598e+00	 1.2097389e-01	 1.6110247e+00	 1.2292532e-01	  1.2670340e+00 	 1.9191837e-01
     146	 1.5592177e+00	 1.2081706e-01	 1.6130291e+00	 1.2246118e-01	  1.2638176e+00 	 1.8753409e-01


.. parsed-literal::

     147	 1.5595536e+00	 1.2075748e-01	 1.6135689e+00	 1.2168404e-01	  1.2636524e+00 	 1.9654727e-01
     148	 1.5614476e+00	 1.2070270e-01	 1.6153601e+00	 1.2181436e-01	  1.2616172e+00 	 1.9628048e-01


.. parsed-literal::

     149	 1.5625213e+00	 1.2065559e-01	 1.6164476e+00	 1.2178945e-01	  1.2563205e+00 	 1.9073844e-01


.. parsed-literal::

     150	 1.5635220e+00	 1.2059618e-01	 1.6174778e+00	 1.2165988e-01	  1.2511712e+00 	 2.0458269e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.05 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff6ec0766e0>



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
    CPU times: user 1.81 s, sys: 42 ms, total: 1.85 s
    Wall time: 576 ms


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

