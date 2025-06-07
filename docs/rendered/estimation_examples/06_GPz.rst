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
       1	-3.4570882e-01	 3.2139608e-01	-3.3608856e-01	 3.1758821e-01	[-3.2925833e-01]	 4.5707941e-01


.. parsed-literal::

       2	-2.7684449e-01	 3.1157411e-01	-2.5394306e-01	 3.0699150e-01	[-2.4057683e-01]	 2.2680306e-01


.. parsed-literal::

       3	-2.3238679e-01	 2.9050366e-01	-1.9072925e-01	 2.8229685e-01	[-1.5983430e-01]	 2.8224492e-01
       4	-1.9357500e-01	 2.6686117e-01	-1.5200856e-01	 2.6014036e-01	[-1.1657437e-01]	 1.7485046e-01


.. parsed-literal::

       5	-1.0587579e-01	 2.5790404e-01	-7.3117114e-02	 2.5303681e-01	[-5.0076278e-02]	 2.0362377e-01
       6	-7.3810137e-02	 2.5278513e-01	-4.4690622e-02	 2.4721742e-01	[-2.4912254e-02]	 2.0423484e-01


.. parsed-literal::

       7	-5.6990947e-02	 2.5015303e-01	-3.3484477e-02	 2.4458168e-01	[-1.2084347e-02]	 1.8927145e-01
       8	-4.4183043e-02	 2.4798494e-01	-2.4444999e-02	 2.4254030e-01	[-2.4871742e-03]	 2.0273709e-01


.. parsed-literal::

       9	-3.1189399e-02	 2.4552857e-01	-1.4059898e-02	 2.4044887e-01	[ 7.6838231e-03]	 1.9687486e-01


.. parsed-literal::

      10	-2.0486771e-02	 2.4338488e-01	-5.3768446e-03	 2.3812746e-01	[ 1.9760384e-02]	 2.1908236e-01


.. parsed-literal::

      11	-1.6349209e-02	 2.4282714e-01	-2.3323513e-03	 2.3820225e-01	  1.8605027e-02 	 2.0959878e-01


.. parsed-literal::

      12	-1.3403473e-02	 2.4236480e-01	 4.1393485e-04	 2.3734364e-01	[ 2.3750944e-02]	 2.1368790e-01


.. parsed-literal::

      13	-1.0395262e-02	 2.4173385e-01	 3.4249744e-03	 2.3633278e-01	[ 2.7947439e-02]	 2.0818949e-01


.. parsed-literal::

      14	-5.5830252e-03	 2.4061217e-01	 9.1955951e-03	 2.3419105e-01	[ 3.7664300e-02]	 2.0555282e-01


.. parsed-literal::

      15	 8.2158342e-02	 2.2804609e-01	 1.0588961e-01	 2.2256900e-01	[ 1.1112424e-01]	 3.2153535e-01


.. parsed-literal::

      16	 2.7201498e-01	 2.1958113e-01	 3.0078550e-01	 2.1454428e-01	[ 3.0919536e-01]	 2.0426106e-01


.. parsed-literal::

      17	 3.0474729e-01	 2.1718844e-01	 3.3577129e-01	 2.1395873e-01	[ 3.4537065e-01]	 3.1232333e-01


.. parsed-literal::

      18	 3.3552884e-01	 2.1272394e-01	 3.6871239e-01	 2.0821929e-01	[ 3.8055918e-01]	 2.1037197e-01
      19	 3.7200762e-01	 2.1119394e-01	 4.0486356e-01	 2.0475361e-01	[ 4.1556094e-01]	 1.8972087e-01


.. parsed-literal::

      20	 4.0990830e-01	 2.0898545e-01	 4.4300601e-01	 2.0114473e-01	[ 4.4478415e-01]	 1.9071388e-01


.. parsed-literal::

      21	 4.7368822e-01	 2.0619053e-01	 5.0828821e-01	 1.9803676e-01	[ 4.9192827e-01]	 2.0582461e-01
      22	 5.5407750e-01	 2.0726856e-01	 5.9230718e-01	 1.9693992e-01	[ 5.2017696e-01]	 2.0500255e-01


.. parsed-literal::

      23	 5.8321185e-01	 2.0533854e-01	 6.2372223e-01	 1.9592010e-01	  5.0624565e-01 	 2.0418143e-01
      24	 6.1802709e-01	 2.0094840e-01	 6.5496674e-01	 1.9241298e-01	[ 5.8144312e-01]	 1.8673086e-01


.. parsed-literal::

      25	 6.5000408e-01	 2.0029065e-01	 6.8778785e-01	 1.9269324e-01	[ 6.0648821e-01]	 1.9504023e-01


.. parsed-literal::

      26	 6.7046734e-01	 2.0116270e-01	 7.0895937e-01	 1.9330692e-01	[ 6.2062905e-01]	 2.1184778e-01


.. parsed-literal::

      27	 7.0428040e-01	 1.9781162e-01	 7.4275389e-01	 1.9039307e-01	[ 6.5231241e-01]	 2.0620942e-01


.. parsed-literal::

      28	 7.4173606e-01	 1.9701504e-01	 7.8027740e-01	 1.8930323e-01	[ 6.8966453e-01]	 2.0874715e-01
      29	 7.8144487e-01	 1.9475728e-01	 8.2054513e-01	 1.8663870e-01	[ 7.3695256e-01]	 1.7663121e-01


.. parsed-literal::

      30	 8.2178298e-01	 1.9872353e-01	 8.6165765e-01	 1.8900014e-01	[ 7.8824266e-01]	 2.0508361e-01
      31	 8.5568995e-01	 1.9672124e-01	 8.9702356e-01	 1.8670248e-01	[ 8.1567680e-01]	 1.7767906e-01


.. parsed-literal::

      32	 8.7911631e-01	 1.9782319e-01	 9.2056299e-01	 1.8774637e-01	[ 8.4248686e-01]	 1.9860506e-01
      33	 8.9531308e-01	 1.9846042e-01	 9.3681517e-01	 1.8790501e-01	[ 8.5984156e-01]	 1.9959402e-01


.. parsed-literal::

      34	 9.1716484e-01	 1.9797849e-01	 9.5900538e-01	 1.8728585e-01	[ 8.8372057e-01]	 2.0918417e-01


.. parsed-literal::

      35	 9.4337911e-01	 1.9794500e-01	 9.8595825e-01	 1.8602252e-01	[ 9.1938545e-01]	 2.0362115e-01


.. parsed-literal::

      36	 9.6949321e-01	 1.9667614e-01	 1.0129431e+00	 1.8514243e-01	[ 9.5229173e-01]	 2.1117258e-01


.. parsed-literal::

      37	 9.8991913e-01	 1.9212613e-01	 1.0347329e+00	 1.8103977e-01	[ 9.7068895e-01]	 2.0134091e-01
      38	 1.0035694e+00	 1.8653953e-01	 1.0496360e+00	 1.7585280e-01	[ 9.8963396e-01]	 1.7258263e-01


.. parsed-literal::

      39	 1.0198430e+00	 1.8345664e-01	 1.0659214e+00	 1.7346287e-01	[ 1.0019245e+00]	 1.7925262e-01


.. parsed-literal::

      40	 1.0339725e+00	 1.8053374e-01	 1.0804457e+00	 1.7160008e-01	[ 1.0127510e+00]	 2.1015954e-01


.. parsed-literal::

      41	 1.0522663e+00	 1.7616759e-01	 1.0988603e+00	 1.6812912e-01	[ 1.0298012e+00]	 2.0846486e-01


.. parsed-literal::

      42	 1.0682220e+00	 1.7168109e-01	 1.1148470e+00	 1.6559523e-01	[ 1.0416527e+00]	 2.1041512e-01
      43	 1.0794792e+00	 1.6983714e-01	 1.1259870e+00	 1.6370568e-01	[ 1.0494352e+00]	 1.9911718e-01


.. parsed-literal::

      44	 1.0963605e+00	 1.6707329e-01	 1.1430156e+00	 1.6113799e-01	[ 1.0559614e+00]	 2.0060277e-01


.. parsed-literal::

      45	 1.1145858e+00	 1.6306274e-01	 1.1615780e+00	 1.5872186e-01	[ 1.0627330e+00]	 2.1125984e-01


.. parsed-literal::

      46	 1.1225502e+00	 1.5849183e-01	 1.1697254e+00	 1.5705704e-01	[ 1.0641655e+00]	 2.0918536e-01


.. parsed-literal::

      47	 1.1428947e+00	 1.5709330e-01	 1.1897620e+00	 1.5697448e-01	[ 1.0778657e+00]	 2.0782638e-01


.. parsed-literal::

      48	 1.1522028e+00	 1.5653302e-01	 1.1992863e+00	 1.5651492e-01	[ 1.0835158e+00]	 2.1123457e-01


.. parsed-literal::

      49	 1.1659812e+00	 1.5456968e-01	 1.2135355e+00	 1.5538484e-01	[ 1.0875914e+00]	 2.0287442e-01
      50	 1.1781094e+00	 1.5346055e-01	 1.2258155e+00	 1.5416527e-01	[ 1.1000454e+00]	 1.8444085e-01


.. parsed-literal::

      51	 1.1921205e+00	 1.5068811e-01	 1.2398227e+00	 1.5199673e-01	[ 1.1130928e+00]	 2.0089769e-01
      52	 1.2017374e+00	 1.4974050e-01	 1.2496708e+00	 1.5091023e-01	[ 1.1213494e+00]	 2.0171022e-01


.. parsed-literal::

      53	 1.2121280e+00	 1.4864187e-01	 1.2607369e+00	 1.4926984e-01	[ 1.1322084e+00]	 2.0488787e-01


.. parsed-literal::

      54	 1.2212766e+00	 1.4806635e-01	 1.2704058e+00	 1.4802316e-01	[ 1.1372610e+00]	 2.1184230e-01


.. parsed-literal::

      55	 1.2324914e+00	 1.4699738e-01	 1.2816861e+00	 1.4748619e-01	[ 1.1455993e+00]	 2.0878887e-01
      56	 1.2454968e+00	 1.4575883e-01	 1.2949775e+00	 1.4688515e-01	[ 1.1490000e+00]	 1.8179560e-01


.. parsed-literal::

      57	 1.2535329e+00	 1.4536618e-01	 1.3031893e+00	 1.4680909e-01	[ 1.1490753e+00]	 2.1420264e-01
      58	 1.2628531e+00	 1.4399490e-01	 1.3125396e+00	 1.4615459e-01	[ 1.1529382e+00]	 2.0002651e-01


.. parsed-literal::

      59	 1.2726392e+00	 1.4243373e-01	 1.3223949e+00	 1.4484292e-01	[ 1.1577082e+00]	 2.1251130e-01


.. parsed-literal::

      60	 1.2809851e+00	 1.4152850e-01	 1.3309167e+00	 1.4398707e-01	[ 1.1677180e+00]	 2.0513153e-01
      61	 1.2898069e+00	 1.4071880e-01	 1.3398259e+00	 1.4309354e-01	[ 1.1784104e+00]	 1.9968605e-01


.. parsed-literal::

      62	 1.2976523e+00	 1.4065578e-01	 1.3475162e+00	 1.4277125e-01	[ 1.2019043e+00]	 2.0019937e-01


.. parsed-literal::

      63	 1.3048013e+00	 1.4051196e-01	 1.3546733e+00	 1.4267270e-01	[ 1.2099834e+00]	 2.0958161e-01


.. parsed-literal::

      64	 1.3152742e+00	 1.3994768e-01	 1.3651878e+00	 1.4232304e-01	[ 1.2199909e+00]	 2.1571493e-01


.. parsed-literal::

      65	 1.3192249e+00	 1.3868632e-01	 1.3694217e+00	 1.4156226e-01	[ 1.2237805e+00]	 2.1779752e-01


.. parsed-literal::

      66	 1.3314163e+00	 1.3801090e-01	 1.3813439e+00	 1.4147784e-01	[ 1.2292466e+00]	 2.1323895e-01


.. parsed-literal::

      67	 1.3351563e+00	 1.3743402e-01	 1.3850312e+00	 1.4101057e-01	[ 1.2310442e+00]	 2.1840572e-01
      68	 1.3418953e+00	 1.3623994e-01	 1.3919692e+00	 1.4013933e-01	[ 1.2337580e+00]	 1.7856407e-01


.. parsed-literal::

      69	 1.3508887e+00	 1.3502781e-01	 1.4011547e+00	 1.3932468e-01	[ 1.2396502e+00]	 2.0076227e-01


.. parsed-literal::

      70	 1.3582819e+00	 1.3339418e-01	 1.4089486e+00	 1.3835323e-01	[ 1.2531068e+00]	 2.1676254e-01
      71	 1.3663094e+00	 1.3319154e-01	 1.4169506e+00	 1.3826499e-01	[ 1.2551235e+00]	 1.7299509e-01


.. parsed-literal::

      72	 1.3720075e+00	 1.3304652e-01	 1.4226813e+00	 1.3831423e-01	[ 1.2568404e+00]	 2.0519781e-01
      73	 1.3761444e+00	 1.3231743e-01	 1.4271388e+00	 1.3813002e-01	  1.2559312e+00 	 1.7482662e-01


.. parsed-literal::

      74	 1.3812261e+00	 1.3202025e-01	 1.4323122e+00	 1.3814079e-01	[ 1.2577451e+00]	 2.1503735e-01


.. parsed-literal::

      75	 1.3849900e+00	 1.3156500e-01	 1.4362543e+00	 1.3805367e-01	[ 1.2581522e+00]	 2.0188618e-01


.. parsed-literal::

      76	 1.3894843e+00	 1.3109034e-01	 1.4408913e+00	 1.3796780e-01	[ 1.2587701e+00]	 2.0348096e-01
      77	 1.3966335e+00	 1.3010067e-01	 1.4482352e+00	 1.3738619e-01	[ 1.2650185e+00]	 1.9578385e-01


.. parsed-literal::

      78	 1.3994371e+00	 1.3032115e-01	 1.4511877e+00	 1.3826836e-01	  1.2588053e+00 	 1.9755101e-01


.. parsed-literal::

      79	 1.4048925e+00	 1.2993200e-01	 1.4562958e+00	 1.3758753e-01	[ 1.2692365e+00]	 2.1606874e-01


.. parsed-literal::

      80	 1.4080056e+00	 1.2971157e-01	 1.4593725e+00	 1.3759329e-01	[ 1.2719439e+00]	 2.0155287e-01


.. parsed-literal::

      81	 1.4126898e+00	 1.2944025e-01	 1.4640262e+00	 1.3767424e-01	[ 1.2727789e+00]	 2.0773935e-01


.. parsed-literal::

      82	 1.4171675e+00	 1.2919622e-01	 1.4686416e+00	 1.3836152e-01	  1.2656242e+00 	 2.0290589e-01
      83	 1.4224291e+00	 1.2906114e-01	 1.4738819e+00	 1.3838255e-01	  1.2664379e+00 	 1.8101430e-01


.. parsed-literal::

      84	 1.4261608e+00	 1.2884187e-01	 1.4776601e+00	 1.3810993e-01	  1.2674029e+00 	 2.0912170e-01


.. parsed-literal::

      85	 1.4295031e+00	 1.2857186e-01	 1.4810998e+00	 1.3774788e-01	  1.2665346e+00 	 2.0194578e-01
      86	 1.4343173e+00	 1.2781370e-01	 1.4861312e+00	 1.3650766e-01	  1.2681188e+00 	 1.7716718e-01


.. parsed-literal::

      87	 1.4384033e+00	 1.2752599e-01	 1.4902459e+00	 1.3614227e-01	  1.2723806e+00 	 1.9559312e-01


.. parsed-literal::

      88	 1.4416104e+00	 1.2727160e-01	 1.4934862e+00	 1.3599078e-01	[ 1.2782362e+00]	 2.0564175e-01


.. parsed-literal::

      89	 1.4444554e+00	 1.2702217e-01	 1.4963688e+00	 1.3570937e-01	[ 1.2833929e+00]	 2.0767975e-01


.. parsed-literal::

      90	 1.4480624e+00	 1.2676539e-01	 1.5000549e+00	 1.3548989e-01	[ 1.2876826e+00]	 2.1613836e-01


.. parsed-literal::

      91	 1.4514901e+00	 1.2650780e-01	 1.5035956e+00	 1.3524491e-01	[ 1.2897519e+00]	 2.0083809e-01
      92	 1.4550280e+00	 1.2613071e-01	 1.5073100e+00	 1.3479712e-01	  1.2891867e+00 	 1.9869232e-01


.. parsed-literal::

      93	 1.4584823e+00	 1.2581776e-01	 1.5107693e+00	 1.3451151e-01	[ 1.2902024e+00]	 2.0336032e-01


.. parsed-literal::

      94	 1.4608144e+00	 1.2560629e-01	 1.5130828e+00	 1.3432598e-01	  1.2899208e+00 	 2.0692420e-01
      95	 1.4648260e+00	 1.2527254e-01	 1.5170641e+00	 1.3381290e-01	[ 1.2913437e+00]	 1.8487978e-01


.. parsed-literal::

      96	 1.4657313e+00	 1.2499211e-01	 1.5180476e+00	 1.3377885e-01	[ 1.2929258e+00]	 2.1111774e-01


.. parsed-literal::

      97	 1.4699197e+00	 1.2502280e-01	 1.5220979e+00	 1.3356570e-01	[ 1.2980635e+00]	 2.1031594e-01


.. parsed-literal::

      98	 1.4716563e+00	 1.2504636e-01	 1.5238274e+00	 1.3346992e-01	[ 1.3008134e+00]	 2.0614648e-01
      99	 1.4740300e+00	 1.2507065e-01	 1.5262569e+00	 1.3343304e-01	[ 1.3040805e+00]	 1.9827580e-01


.. parsed-literal::

     100	 1.4751575e+00	 1.2496040e-01	 1.5276346e+00	 1.3311148e-01	  1.3034800e+00 	 2.1474910e-01
     101	 1.4781445e+00	 1.2492150e-01	 1.5305312e+00	 1.3323456e-01	[ 1.3060998e+00]	 1.8515015e-01


.. parsed-literal::

     102	 1.4800111e+00	 1.2481408e-01	 1.5324340e+00	 1.3324732e-01	  1.3060769e+00 	 1.8933606e-01
     103	 1.4818899e+00	 1.2463478e-01	 1.5343928e+00	 1.3313432e-01	  1.3050802e+00 	 1.9406033e-01


.. parsed-literal::

     104	 1.4849419e+00	 1.2435532e-01	 1.5375234e+00	 1.3278326e-01	  1.3050606e+00 	 2.0346975e-01


.. parsed-literal::

     105	 1.4870319e+00	 1.2402168e-01	 1.5397008e+00	 1.3225855e-01	  1.3031582e+00 	 3.2855797e-01


.. parsed-literal::

     106	 1.4904195e+00	 1.2379481e-01	 1.5431295e+00	 1.3172551e-01	  1.3035566e+00 	 2.1274590e-01
     107	 1.4926648e+00	 1.2368533e-01	 1.5453697e+00	 1.3141192e-01	  1.3035945e+00 	 1.8455696e-01


.. parsed-literal::

     108	 1.4953458e+00	 1.2358024e-01	 1.5481402e+00	 1.3097851e-01	  1.3024874e+00 	 2.1376705e-01


.. parsed-literal::

     109	 1.4975869e+00	 1.2350591e-01	 1.5504947e+00	 1.3091304e-01	  1.2972281e+00 	 2.0412159e-01


.. parsed-literal::

     110	 1.4995594e+00	 1.2333931e-01	 1.5525024e+00	 1.3088562e-01	  1.2958232e+00 	 2.1492410e-01


.. parsed-literal::

     111	 1.5019168e+00	 1.2315213e-01	 1.5549893e+00	 1.3087359e-01	  1.2912246e+00 	 2.1636939e-01
     112	 1.5040730e+00	 1.2289497e-01	 1.5572136e+00	 1.3082509e-01	  1.2913000e+00 	 1.7828083e-01


.. parsed-literal::

     113	 1.5068915e+00	 1.2268006e-01	 1.5601420e+00	 1.3071443e-01	  1.2896792e+00 	 2.0403695e-01


.. parsed-literal::

     114	 1.5089004e+00	 1.2241438e-01	 1.5621071e+00	 1.3064840e-01	  1.2956261e+00 	 2.0804143e-01


.. parsed-literal::

     115	 1.5106379e+00	 1.2215213e-01	 1.5637791e+00	 1.3033575e-01	  1.2992041e+00 	 2.0177102e-01


.. parsed-literal::

     116	 1.5122420e+00	 1.2201659e-01	 1.5653551e+00	 1.3024438e-01	  1.3013215e+00 	 2.2394490e-01
     117	 1.5146545e+00	 1.2166539e-01	 1.5678377e+00	 1.3017736e-01	  1.2990839e+00 	 1.9823527e-01


.. parsed-literal::

     118	 1.5170760e+00	 1.2149656e-01	 1.5703457e+00	 1.3045017e-01	  1.2991822e+00 	 2.1314645e-01


.. parsed-literal::

     119	 1.5188965e+00	 1.2139617e-01	 1.5722358e+00	 1.3060481e-01	  1.2968000e+00 	 2.1452498e-01
     120	 1.5206171e+00	 1.2128105e-01	 1.5741538e+00	 1.3095697e-01	  1.2924019e+00 	 1.9813299e-01


.. parsed-literal::

     121	 1.5223305e+00	 1.2121938e-01	 1.5758660e+00	 1.3080908e-01	  1.2922199e+00 	 1.8819237e-01
     122	 1.5236428e+00	 1.2111789e-01	 1.5771644e+00	 1.3063038e-01	  1.2930483e+00 	 1.8115234e-01


.. parsed-literal::

     123	 1.5262308e+00	 1.2079967e-01	 1.5797530e+00	 1.3018673e-01	  1.2936323e+00 	 2.1489263e-01


.. parsed-literal::

     124	 1.5272996e+00	 1.2078551e-01	 1.5808604e+00	 1.3018313e-01	  1.2910214e+00 	 3.0977392e-01


.. parsed-literal::

     125	 1.5287332e+00	 1.2068379e-01	 1.5822625e+00	 1.3007754e-01	  1.2917665e+00 	 2.1614623e-01


.. parsed-literal::

     126	 1.5307831e+00	 1.2053950e-01	 1.5843106e+00	 1.3005938e-01	  1.2898478e+00 	 2.1131015e-01


.. parsed-literal::

     127	 1.5322221e+00	 1.2046706e-01	 1.5857565e+00	 1.3010822e-01	  1.2892557e+00 	 2.0660043e-01


.. parsed-literal::

     128	 1.5338251e+00	 1.2031655e-01	 1.5874705e+00	 1.3015542e-01	  1.2817589e+00 	 2.1662736e-01


.. parsed-literal::

     129	 1.5351916e+00	 1.2027606e-01	 1.5888611e+00	 1.3029315e-01	  1.2812982e+00 	 2.1674871e-01
     130	 1.5365537e+00	 1.2018362e-01	 1.5902330e+00	 1.3025056e-01	  1.2812459e+00 	 1.7801023e-01


.. parsed-literal::

     131	 1.5383083e+00	 1.2009927e-01	 1.5920141e+00	 1.3024519e-01	  1.2810403e+00 	 2.0216894e-01
     132	 1.5393692e+00	 1.1991692e-01	 1.5931444e+00	 1.3020793e-01	  1.2779734e+00 	 1.7743444e-01


.. parsed-literal::

     133	 1.5407867e+00	 1.1991477e-01	 1.5944935e+00	 1.3021096e-01	  1.2815528e+00 	 2.0574617e-01


.. parsed-literal::

     134	 1.5415435e+00	 1.1988173e-01	 1.5952261e+00	 1.3022321e-01	  1.2825935e+00 	 2.0721722e-01


.. parsed-literal::

     135	 1.5427820e+00	 1.1975888e-01	 1.5964459e+00	 1.3018203e-01	  1.2846942e+00 	 2.1576667e-01


.. parsed-literal::

     136	 1.5438902e+00	 1.1954054e-01	 1.5975943e+00	 1.3029165e-01	  1.2845548e+00 	 2.0547795e-01


.. parsed-literal::

     137	 1.5455029e+00	 1.1936573e-01	 1.5991870e+00	 1.3017335e-01	  1.2869953e+00 	 2.0077920e-01


.. parsed-literal::

     138	 1.5464230e+00	 1.1925237e-01	 1.6001327e+00	 1.3015257e-01	  1.2859480e+00 	 2.1924376e-01


.. parsed-literal::

     139	 1.5479097e+00	 1.1904387e-01	 1.6017041e+00	 1.3014192e-01	  1.2827095e+00 	 2.1363354e-01


.. parsed-literal::

     140	 1.5488237e+00	 1.1886865e-01	 1.6027667e+00	 1.3032891e-01	  1.2782639e+00 	 2.0576835e-01


.. parsed-literal::

     141	 1.5501228e+00	 1.1878264e-01	 1.6040771e+00	 1.3024188e-01	  1.2765766e+00 	 2.1234322e-01
     142	 1.5512179e+00	 1.1875879e-01	 1.6051910e+00	 1.3020799e-01	  1.2761543e+00 	 2.0530319e-01


.. parsed-literal::

     143	 1.5522101e+00	 1.1869686e-01	 1.6062227e+00	 1.3014680e-01	  1.2747253e+00 	 1.7586923e-01


.. parsed-literal::

     144	 1.5535850e+00	 1.1863555e-01	 1.6077038e+00	 1.3022317e-01	  1.2726087e+00 	 2.1297479e-01
     145	 1.5551455e+00	 1.1844927e-01	 1.6092914e+00	 1.2999638e-01	  1.2699737e+00 	 1.8718505e-01


.. parsed-literal::

     146	 1.5560709e+00	 1.1835655e-01	 1.6102071e+00	 1.2995277e-01	  1.2707815e+00 	 1.7655706e-01
     147	 1.5575146e+00	 1.1822916e-01	 1.6116226e+00	 1.2993082e-01	  1.2715628e+00 	 1.8643284e-01


.. parsed-literal::

     148	 1.5583109e+00	 1.1815915e-01	 1.6124323e+00	 1.3000865e-01	  1.2728208e+00 	 1.9662857e-01
     149	 1.5595464e+00	 1.1813624e-01	 1.6136063e+00	 1.2999174e-01	  1.2734678e+00 	 1.9668174e-01


.. parsed-literal::

     150	 1.5602617e+00	 1.1812215e-01	 1.6143065e+00	 1.2997860e-01	  1.2738151e+00 	 1.9956064e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.06 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5d2b4f3dc0>



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


.. parsed-literal::

    CPU times: user 1.76 s, sys: 42.9 ms, total: 1.81 s
    Wall time: 592 ms


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

