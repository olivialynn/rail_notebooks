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
       1	-3.6048894e-01	 3.2561270e-01	-3.5076925e-01	 2.9898537e-01	[-3.0389794e-01]	 4.6745300e-01


.. parsed-literal::

       2	-2.8482741e-01	 3.1318997e-01	-2.5930126e-01	 2.8923447e-01	[-1.9106715e-01]	 2.2844863e-01


.. parsed-literal::

       3	-2.4009247e-01	 2.9244318e-01	-1.9676052e-01	 2.7505950e-01	[-1.2905969e-01]	 2.3540783e-01


.. parsed-literal::

       4	-1.9678880e-01	 2.7328005e-01	-1.4630530e-01	 2.6354593e-01	[-9.8302772e-02]	 2.3519373e-01
       5	-1.4723799e-01	 2.6028401e-01	-1.1762920e-01	 2.4874873e-01	[-5.3757983e-02]	 1.1554813e-01


.. parsed-literal::

       6	-8.3887061e-02	 2.5606546e-01	-5.6272818e-02	 2.4375061e-01	[-1.0424151e-02]	 1.1814570e-01
       7	-6.5149143e-02	 2.5195210e-01	-4.1523537e-02	 2.3916912e-01	[ 4.3610768e-03]	 1.1452723e-01


.. parsed-literal::

       8	-5.3647025e-02	 2.5004008e-01	-3.2917525e-02	 2.3725135e-01	[ 1.5600111e-02]	 1.1692858e-01
       9	-3.6495576e-02	 2.4677722e-01	-1.9164964e-02	 2.3478555e-01	[ 2.6721635e-02]	 1.2431622e-01


.. parsed-literal::

      10	-2.5655098e-02	 2.4463478e-01	-1.0289781e-02	 2.3384781e-01	[ 3.3833904e-02]	 1.1771441e-01
      11	-2.0025944e-02	 2.4365646e-01	-5.5567237e-03	 2.3331648e-01	[ 3.5739883e-02]	 1.1592579e-01


.. parsed-literal::

      12	-1.6234234e-02	 2.4290545e-01	-2.0777330e-03	 2.3307395e-01	[ 3.6364645e-02]	 1.1799979e-01
      13	-1.1577789e-02	 2.4196136e-01	 2.5152933e-03	 2.3260811e-01	[ 3.8892764e-02]	 1.1578798e-01


.. parsed-literal::

      14	 6.2819124e-02	 2.2891056e-01	 8.1868060e-02	 2.2198267e-01	[ 1.0492987e-01]	 2.3402953e-01


.. parsed-literal::

      15	 7.9360121e-02	 2.2633439e-01	 1.0059280e-01	 2.2030377e-01	[ 1.1284109e-01]	 2.3310232e-01
      16	 1.6656657e-01	 2.2025941e-01	 1.9131519e-01	 2.1736965e-01	[ 1.9924315e-01]	 1.1861587e-01


.. parsed-literal::

      17	 2.0689294e-01	 2.1645650e-01	 2.3876981e-01	 2.1289739e-01	[ 2.5826810e-01]	 1.1742306e-01
      18	 2.9354253e-01	 2.1343237e-01	 3.2527383e-01	 2.1155372e-01	[ 3.3459640e-01]	 1.1567163e-01


.. parsed-literal::

      19	 3.2186577e-01	 2.1057400e-01	 3.5370186e-01	 2.1103373e-01	[ 3.5922773e-01]	 1.1761236e-01
      20	 3.6368494e-01	 2.0736018e-01	 3.9712137e-01	 2.0826143e-01	[ 4.0346076e-01]	 1.1687684e-01


.. parsed-literal::

      21	 4.6913185e-01	 2.0641034e-01	 5.0237928e-01	 2.0828896e-01	[ 5.1010292e-01]	 1.1869621e-01
      22	 5.5733979e-01	 2.0521633e-01	 5.9396630e-01	 2.0719974e-01	[ 6.0031200e-01]	 1.1648464e-01


.. parsed-literal::

      23	 6.2753494e-01	 1.9546303e-01	 6.6671750e-01	 2.0081820e-01	[ 6.7398445e-01]	 1.1757183e-01
      24	 6.6539586e-01	 1.9191619e-01	 7.0392709e-01	 1.9550958e-01	[ 7.1461286e-01]	 1.2547016e-01


.. parsed-literal::

      25	 7.0005832e-01	 1.8978489e-01	 7.3825314e-01	 1.9090836e-01	[ 7.5338022e-01]	 1.1827230e-01
      26	 7.4736094e-01	 1.8476234e-01	 7.8601634e-01	 1.8468220e-01	[ 7.9563927e-01]	 1.1545610e-01


.. parsed-literal::

      27	 7.8878865e-01	 1.8926618e-01	 8.2795370e-01	 1.8656364e-01	[ 8.3267859e-01]	 1.1756206e-01
      28	 8.1918417e-01	 1.8953535e-01	 8.5874423e-01	 1.8614007e-01	[ 8.6375263e-01]	 1.3389635e-01


.. parsed-literal::

      29	 8.4775776e-01	 1.8998667e-01	 8.8815423e-01	 1.8759652e-01	[ 8.9193765e-01]	 1.1808491e-01
      30	 8.7967937e-01	 1.8593375e-01	 9.2095600e-01	 1.8463816e-01	[ 9.2818045e-01]	 1.1601996e-01


.. parsed-literal::

      31	 9.0591878e-01	 1.8089037e-01	 9.4699535e-01	 1.8312731e-01	[ 9.5172474e-01]	 1.1794257e-01
      32	 9.2705503e-01	 1.7531594e-01	 9.6898139e-01	 1.7605004e-01	[ 9.7167446e-01]	 1.1705375e-01


.. parsed-literal::

      33	 9.5204593e-01	 1.7134755e-01	 9.9422424e-01	 1.7279622e-01	[ 1.0012532e+00]	 1.2672710e-01
      34	 9.7697846e-01	 1.6693168e-01	 1.0199808e+00	 1.6774839e-01	[ 1.0216978e+00]	 1.1611509e-01


.. parsed-literal::

      35	 9.9989012e-01	 1.6430601e-01	 1.0443368e+00	 1.6471341e-01	[ 1.0419711e+00]	 1.1862493e-01
      36	 1.0150675e+00	 1.6291809e-01	 1.0605633e+00	 1.6183365e-01	[ 1.0573412e+00]	 1.1515164e-01


.. parsed-literal::

      37	 1.0287946e+00	 1.6141212e-01	 1.0744258e+00	 1.6106212e-01	[ 1.0678673e+00]	 1.1769795e-01
      38	 1.0430788e+00	 1.5936292e-01	 1.0888148e+00	 1.5983353e-01	[ 1.0823212e+00]	 1.1585355e-01


.. parsed-literal::

      39	 1.0626851e+00	 1.5662796e-01	 1.1087978e+00	 1.5713256e-01	[ 1.0981796e+00]	 1.1896706e-01
      40	 1.0730757e+00	 1.5608103e-01	 1.1198223e+00	 1.5771971e-01	[ 1.1063375e+00]	 1.1644626e-01


.. parsed-literal::

      41	 1.0855395e+00	 1.5537477e-01	 1.1316998e+00	 1.5687513e-01	[ 1.1142927e+00]	 1.2683797e-01
      42	 1.0950072e+00	 1.5435788e-01	 1.1412940e+00	 1.5475041e-01	[ 1.1209123e+00]	 1.1513543e-01


.. parsed-literal::

      43	 1.1102623e+00	 1.5265958e-01	 1.1568798e+00	 1.5105722e-01	[ 1.1293176e+00]	 1.2121415e-01
      44	 1.1272274e+00	 1.5045319e-01	 1.1739036e+00	 1.4629658e-01	[ 1.1471646e+00]	 1.1605954e-01


.. parsed-literal::

      45	 1.1412136e+00	 1.4921375e-01	 1.1879843e+00	 1.4400976e-01	[ 1.1572029e+00]	 1.1809587e-01
      46	 1.1521913e+00	 1.4837723e-01	 1.1990758e+00	 1.4286485e-01	[ 1.1697809e+00]	 1.1556172e-01


.. parsed-literal::

      47	 1.1653081e+00	 1.4756651e-01	 1.2128638e+00	 1.4108540e-01	[ 1.1824730e+00]	 1.1757231e-01
      48	 1.1737932e+00	 1.4649606e-01	 1.2219401e+00	 1.3993859e-01	[ 1.1963409e+00]	 1.1543870e-01


.. parsed-literal::

      49	 1.1819918e+00	 1.4606078e-01	 1.2300513e+00	 1.3973976e-01	[ 1.2009548e+00]	 1.1753106e-01
      50	 1.1929124e+00	 1.4504652e-01	 1.2412030e+00	 1.3892580e-01	[ 1.2042142e+00]	 1.2159181e-01


.. parsed-literal::

      51	 1.2028337e+00	 1.4409660e-01	 1.2512526e+00	 1.3840020e-01	[ 1.2105807e+00]	 1.1793995e-01
      52	 1.2212405e+00	 1.4245868e-01	 1.2703419e+00	 1.3711752e-01	[ 1.2205131e+00]	 1.1552763e-01


.. parsed-literal::

      53	 1.2239573e+00	 1.4132806e-01	 1.2735911e+00	 1.3711038e-01	[ 1.2248316e+00]	 1.1833787e-01
      54	 1.2414678e+00	 1.4062953e-01	 1.2902856e+00	 1.3637636e-01	[ 1.2401430e+00]	 1.1612892e-01


.. parsed-literal::

      55	 1.2481526e+00	 1.4030292e-01	 1.2971008e+00	 1.3637423e-01	[ 1.2423407e+00]	 1.1781263e-01
      56	 1.2583006e+00	 1.3949163e-01	 1.3075385e+00	 1.3638098e-01	[ 1.2470186e+00]	 1.1667800e-01


.. parsed-literal::

      57	 1.2684011e+00	 1.3847151e-01	 1.3179468e+00	 1.3586254e-01	[ 1.2523055e+00]	 1.1724138e-01
      58	 1.2799566e+00	 1.3678415e-01	 1.3299179e+00	 1.3473179e-01	[ 1.2626411e+00]	 1.2553740e-01


.. parsed-literal::

      59	 1.2887855e+00	 1.3633335e-01	 1.3386383e+00	 1.3406272e-01	[ 1.2711036e+00]	 1.1752248e-01
      60	 1.2961966e+00	 1.3602838e-01	 1.3460729e+00	 1.3352864e-01	[ 1.2790644e+00]	 1.1616373e-01


.. parsed-literal::

      61	 1.3032255e+00	 1.3567620e-01	 1.3532902e+00	 1.3351981e-01	[ 1.2840884e+00]	 1.1723542e-01
      62	 1.3111409e+00	 1.3496193e-01	 1.3612045e+00	 1.3286231e-01	[ 1.2961430e+00]	 1.1616015e-01


.. parsed-literal::

      63	 1.3197160e+00	 1.3437345e-01	 1.3699259e+00	 1.3277675e-01	[ 1.3016826e+00]	 1.1822891e-01
      64	 1.3295973e+00	 1.3363832e-01	 1.3800568e+00	 1.3242491e-01	[ 1.3117253e+00]	 1.1571670e-01


.. parsed-literal::

      65	 1.3355666e+00	 1.3334465e-01	 1.3862972e+00	 1.3232056e-01	  1.3080565e+00 	 1.1820102e-01
      66	 1.3429697e+00	 1.3323681e-01	 1.3935211e+00	 1.3209205e-01	[ 1.3163286e+00]	 1.1548018e-01


.. parsed-literal::

      67	 1.3494599e+00	 1.3288211e-01	 1.4000282e+00	 1.3178436e-01	[ 1.3207808e+00]	 1.2842202e-01
      68	 1.3556942e+00	 1.3276809e-01	 1.4064406e+00	 1.3161522e-01	[ 1.3209834e+00]	 1.1646199e-01


.. parsed-literal::

      69	 1.3618898e+00	 1.3181378e-01	 1.4130664e+00	 1.3104679e-01	  1.3209346e+00 	 1.1738181e-01
      70	 1.3704516e+00	 1.3145792e-01	 1.4215049e+00	 1.3082400e-01	[ 1.3287887e+00]	 1.1634207e-01


.. parsed-literal::

      71	 1.3764056e+00	 1.3079783e-01	 1.4275843e+00	 1.3059618e-01	[ 1.3324646e+00]	 1.1816907e-01
      72	 1.3811133e+00	 1.3011384e-01	 1.4326927e+00	 1.3041763e-01	  1.3304590e+00 	 1.1544776e-01


.. parsed-literal::

      73	 1.3862939e+00	 1.2944551e-01	 1.4379155e+00	 1.3006720e-01	[ 1.3342303e+00]	 1.1846018e-01
      74	 1.3920966e+00	 1.2888443e-01	 1.4437829e+00	 1.2975912e-01	[ 1.3358654e+00]	 1.1553216e-01


.. parsed-literal::

      75	 1.3991891e+00	 1.2839913e-01	 1.4509009e+00	 1.2943408e-01	  1.3352171e+00 	 1.2100339e-01
      76	 1.4050099e+00	 1.2767902e-01	 1.4568089e+00	 1.2901860e-01	  1.3290781e+00 	 1.1672449e-01


.. parsed-literal::

      77	 1.4107168e+00	 1.2762638e-01	 1.4623040e+00	 1.2876560e-01	  1.3358411e+00 	 1.1775112e-01
      78	 1.4143317e+00	 1.2747363e-01	 1.4659076e+00	 1.2853308e-01	[ 1.3367586e+00]	 1.1675072e-01


.. parsed-literal::

      79	 1.4180807e+00	 1.2728359e-01	 1.4697599e+00	 1.2815121e-01	[ 1.3396772e+00]	 1.1783338e-01
      80	 1.4220019e+00	 1.2707290e-01	 1.4737480e+00	 1.2787500e-01	[ 1.3398384e+00]	 1.1651897e-01


.. parsed-literal::

      81	 1.4263478e+00	 1.2682655e-01	 1.4781706e+00	 1.2771603e-01	  1.3386979e+00 	 1.1737919e-01
      82	 1.4324120e+00	 1.2650834e-01	 1.4844060e+00	 1.2717133e-01	[ 1.3419590e+00]	 1.1653447e-01


.. parsed-literal::

      83	 1.4361875e+00	 1.2639616e-01	 1.4881464e+00	 1.2726728e-01	  1.3391936e+00 	 1.1853170e-01
      84	 1.4387321e+00	 1.2624655e-01	 1.4906216e+00	 1.2717298e-01	  1.3411150e+00 	 1.2472582e-01


.. parsed-literal::

      85	 1.4427551e+00	 1.2607951e-01	 1.4947240e+00	 1.2690027e-01	  1.3416952e+00 	 1.1859250e-01
      86	 1.4465766e+00	 1.2590907e-01	 1.4987403e+00	 1.2659976e-01	  1.3374839e+00 	 1.1597729e-01


.. parsed-literal::

      87	 1.4508068e+00	 1.2569261e-01	 1.5032386e+00	 1.2613421e-01	  1.3391437e+00 	 1.2090588e-01
      88	 1.4547660e+00	 1.2551885e-01	 1.5072526e+00	 1.2572339e-01	  1.3410169e+00 	 1.1579084e-01


.. parsed-literal::

      89	 1.4579842e+00	 1.2525661e-01	 1.5104210e+00	 1.2553082e-01	[ 1.3441592e+00]	 1.1762094e-01
      90	 1.4611410e+00	 1.2501867e-01	 1.5135379e+00	 1.2517994e-01	[ 1.3495227e+00]	 1.1596656e-01


.. parsed-literal::

      91	 1.4661650e+00	 1.2452745e-01	 1.5185361e+00	 1.2445762e-01	[ 1.3542456e+00]	 1.1702847e-01
      92	 1.4686439e+00	 1.2443551e-01	 1.5210772e+00	 1.2418422e-01	  1.3526348e+00 	 1.1638618e-01


.. parsed-literal::

      93	 1.4711930e+00	 1.2438278e-01	 1.5235191e+00	 1.2405669e-01	[ 1.3580924e+00]	 1.3273931e-01
      94	 1.4730422e+00	 1.2435768e-01	 1.5253498e+00	 1.2400127e-01	[ 1.3595337e+00]	 1.1586761e-01


.. parsed-literal::

      95	 1.4760100e+00	 1.2425982e-01	 1.5283604e+00	 1.2390338e-01	[ 1.3613663e+00]	 1.1798072e-01
      96	 1.4794085e+00	 1.2411148e-01	 1.5319087e+00	 1.2377689e-01	  1.3583044e+00 	 1.1566806e-01


.. parsed-literal::

      97	 1.4828388e+00	 1.2402598e-01	 1.5354344e+00	 1.2358590e-01	[ 1.3621111e+00]	 1.1750722e-01
      98	 1.4850101e+00	 1.2389005e-01	 1.5375902e+00	 1.2342911e-01	[ 1.3637598e+00]	 1.1554766e-01


.. parsed-literal::

      99	 1.4872810e+00	 1.2373459e-01	 1.5399015e+00	 1.2317775e-01	[ 1.3645224e+00]	 1.1745191e-01
     100	 1.4894311e+00	 1.2360184e-01	 1.5421701e+00	 1.2273838e-01	[ 1.3651525e+00]	 1.1531377e-01


.. parsed-literal::

     101	 1.4918578e+00	 1.2350990e-01	 1.5446237e+00	 1.2252167e-01	[ 1.3670996e+00]	 1.2663221e-01
     102	 1.4946728e+00	 1.2340816e-01	 1.5475215e+00	 1.2230740e-01	[ 1.3676475e+00]	 1.1602831e-01


.. parsed-literal::

     103	 1.4965839e+00	 1.2336008e-01	 1.5494830e+00	 1.2209143e-01	[ 1.3688737e+00]	 1.1833048e-01
     104	 1.4992135e+00	 1.2316932e-01	 1.5521527e+00	 1.2194321e-01	[ 1.3690039e+00]	 1.1626983e-01


.. parsed-literal::

     105	 1.5023800e+00	 1.2289510e-01	 1.5553540e+00	 1.2170196e-01	  1.3683229e+00 	 1.1747479e-01
     106	 1.5045801e+00	 1.2245806e-01	 1.5575878e+00	 1.2145808e-01	  1.3631489e+00 	 1.1616349e-01


.. parsed-literal::

     107	 1.5070862e+00	 1.2233845e-01	 1.5600143e+00	 1.2140188e-01	  1.3635293e+00 	 1.1822557e-01
     108	 1.5089104e+00	 1.2226164e-01	 1.5617972e+00	 1.2133732e-01	  1.3609227e+00 	 1.1486149e-01


.. parsed-literal::

     109	 1.5108474e+00	 1.2206597e-01	 1.5637134e+00	 1.2124337e-01	  1.3597045e+00 	 1.1911464e-01
     110	 1.5126877e+00	 1.2191667e-01	 1.5656084e+00	 1.2114118e-01	  1.3535568e+00 	 1.2446284e-01


.. parsed-literal::

     111	 1.5143342e+00	 1.2177324e-01	 1.5672547e+00	 1.2108796e-01	  1.3538647e+00 	 1.1861801e-01
     112	 1.5162323e+00	 1.2147611e-01	 1.5691970e+00	 1.2101351e-01	  1.3557572e+00 	 1.1551762e-01


.. parsed-literal::

     113	 1.5179126e+00	 1.2127779e-01	 1.5709033e+00	 1.2102012e-01	  1.3545310e+00 	 1.1831403e-01
     114	 1.5207373e+00	 1.2081023e-01	 1.5738010e+00	 1.2103215e-01	  1.3543516e+00 	 1.1629510e-01


.. parsed-literal::

     115	 1.5224845e+00	 1.2073329e-01	 1.5755489e+00	 1.2098930e-01	  1.3523720e+00 	 1.1823487e-01
     116	 1.5237560e+00	 1.2073682e-01	 1.5767899e+00	 1.2101040e-01	  1.3520452e+00 	 1.1510801e-01


.. parsed-literal::

     117	 1.5257536e+00	 1.2074340e-01	 1.5787908e+00	 1.2093952e-01	  1.3525085e+00 	 1.5546608e-01
     118	 1.5273338e+00	 1.2055551e-01	 1.5804925e+00	 1.2112060e-01	  1.3444465e+00 	 1.6457033e-01


.. parsed-literal::

     119	 1.5289624e+00	 1.2047510e-01	 1.5821254e+00	 1.2111218e-01	  1.3455102e+00 	 1.1798143e-01
     120	 1.5313638e+00	 1.2031046e-01	 1.5846013e+00	 1.2108844e-01	  1.3434604e+00 	 1.1571884e-01


.. parsed-literal::

     121	 1.5326729e+00	 1.2026686e-01	 1.5859260e+00	 1.2110648e-01	  1.3405987e+00 	 1.1853290e-01
     122	 1.5346977e+00	 1.2027623e-01	 1.5879653e+00	 1.2104554e-01	  1.3376683e+00 	 1.1522317e-01


.. parsed-literal::

     123	 1.5356293e+00	 1.2033692e-01	 1.5888940e+00	 1.2093924e-01	  1.3303917e+00 	 1.1946797e-01
     124	 1.5368540e+00	 1.2033827e-01	 1.5900306e+00	 1.2085346e-01	  1.3359403e+00 	 1.1653876e-01


.. parsed-literal::

     125	 1.5379030e+00	 1.2034674e-01	 1.5910347e+00	 1.2077046e-01	  1.3376386e+00 	 1.1890531e-01
     126	 1.5391411e+00	 1.2035007e-01	 1.5922582e+00	 1.2068350e-01	  1.3373980e+00 	 1.2490869e-01


.. parsed-literal::

     127	 1.5406170e+00	 1.2041346e-01	 1.5937881e+00	 1.2066739e-01	  1.3295770e+00 	 1.2000442e-01
     128	 1.5424044e+00	 1.2041920e-01	 1.5956306e+00	 1.2070900e-01	  1.3290115e+00 	 1.1601019e-01


.. parsed-literal::

     129	 1.5437925e+00	 1.2035911e-01	 1.5970452e+00	 1.2079194e-01	  1.3260120e+00 	 1.1856079e-01
     130	 1.5456846e+00	 1.2028179e-01	 1.5989980e+00	 1.2099976e-01	  1.3287561e+00 	 1.1576223e-01


.. parsed-literal::

     131	 1.5466126e+00	 1.2014472e-01	 1.6000084e+00	 1.2123338e-01	  1.3191878e+00 	 1.1919308e-01
     132	 1.5479386e+00	 1.2015528e-01	 1.6012749e+00	 1.2116297e-01	  1.3255849e+00 	 1.1512733e-01


.. parsed-literal::

     133	 1.5490184e+00	 1.2015338e-01	 1.6023261e+00	 1.2114644e-01	  1.3287677e+00 	 1.1774445e-01
     134	 1.5499210e+00	 1.2010822e-01	 1.6032179e+00	 1.2118398e-01	  1.3281243e+00 	 1.2409711e-01


.. parsed-literal::

     135	 1.5520296e+00	 1.1995480e-01	 1.6053538e+00	 1.2132512e-01	  1.3197751e+00 	 1.1985493e-01


.. parsed-literal::

     136	 1.5531685e+00	 1.1978282e-01	 1.6065209e+00	 1.2150089e-01	  1.3113581e+00 	 2.3318005e-01
     137	 1.5545846e+00	 1.1963363e-01	 1.6079753e+00	 1.2162960e-01	  1.3034569e+00 	 1.1591220e-01


.. parsed-literal::

     138	 1.5561113e+00	 1.1947209e-01	 1.6095733e+00	 1.2175091e-01	  1.2958837e+00 	 1.1884689e-01
     139	 1.5574112e+00	 1.1933497e-01	 1.6109777e+00	 1.2189825e-01	  1.2886361e+00 	 1.1756778e-01


.. parsed-literal::

     140	 1.5585709e+00	 1.1921587e-01	 1.6122138e+00	 1.2195052e-01	  1.2896356e+00 	 1.1936641e-01
     141	 1.5595205e+00	 1.1923946e-01	 1.6131432e+00	 1.2192954e-01	  1.2915517e+00 	 1.1603832e-01


.. parsed-literal::

     142	 1.5606126e+00	 1.1925753e-01	 1.6142156e+00	 1.2191562e-01	  1.2924112e+00 	 1.2752056e-01
     143	 1.5619577e+00	 1.1925713e-01	 1.6155552e+00	 1.2197708e-01	  1.2856101e+00 	 1.2076378e-01


.. parsed-literal::

     144	 1.5637526e+00	 1.1925676e-01	 1.6173319e+00	 1.2212007e-01	  1.2719940e+00 	 1.1837649e-01
     145	 1.5649973e+00	 1.1923940e-01	 1.6185683e+00	 1.2234376e-01	  1.2582991e+00 	 1.1542749e-01


.. parsed-literal::

     146	 1.5661537e+00	 1.1921124e-01	 1.6197061e+00	 1.2241588e-01	  1.2554549e+00 	 1.1963654e-01
     147	 1.5675535e+00	 1.1912724e-01	 1.6211402e+00	 1.2261869e-01	  1.2464597e+00 	 1.3248777e-01


.. parsed-literal::

     148	 1.5685344e+00	 1.1919568e-01	 1.6221514e+00	 1.2273385e-01	  1.2452125e+00 	 1.1857677e-01
     149	 1.5696064e+00	 1.1917650e-01	 1.6232185e+00	 1.2280136e-01	  1.2406694e+00 	 1.1529064e-01


.. parsed-literal::

     150	 1.5710041e+00	 1.1916667e-01	 1.6246640e+00	 1.2295800e-01	  1.2297809e+00 	 1.2770391e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 1min 14s, sys: 1.19 s, total: 1min 15s
    Wall time: 19.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9e95f0cbb0>



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
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.75 s, sys: 47.9 ms, total: 1.8 s
    Wall time: 570 ms


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

