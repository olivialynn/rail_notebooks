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
       1	-3.4637496e-01	 3.2173629e-01	-3.3677192e-01	 3.1559782e-01	[-3.2536835e-01]	 4.5661306e-01


.. parsed-literal::

       2	-2.7959190e-01	 3.1271757e-01	-2.5756502e-01	 3.0620211e-01	[-2.3800480e-01]	 2.3821640e-01


.. parsed-literal::

       3	-2.3545465e-01	 2.9134568e-01	-1.9464151e-01	 2.8040060e-01	[-1.5177758e-01]	 2.8138423e-01
       4	-1.8945332e-01	 2.6808610e-01	-1.4719173e-01	 2.5597622e-01	[-8.3642213e-02]	 1.7621422e-01


.. parsed-literal::

       5	-1.0430428e-01	 2.5771151e-01	-7.4868449e-02	 2.4850131e-01	[-3.2836024e-02]	 2.0859385e-01
       6	-7.4680264e-02	 2.5328669e-01	-4.7409225e-02	 2.4435685e-01	[-1.6311346e-02]	 2.0599461e-01


.. parsed-literal::

       7	-5.7189660e-02	 2.5035816e-01	-3.4379372e-02	 2.4252246e-01	[-4.5489236e-03]	 2.1590877e-01


.. parsed-literal::

       8	-4.3658173e-02	 2.4798562e-01	-2.4385484e-02	 2.4052698e-01	[ 5.5427255e-03]	 2.1766758e-01
       9	-3.2219915e-02	 2.4577898e-01	-1.4909053e-02	 2.3798364e-01	[ 1.5867336e-02]	 1.7782760e-01


.. parsed-literal::

      10	-2.3988049e-02	 2.4380211e-01	-8.8757955e-03	 2.3608851e-01	[ 2.5004485e-02]	 1.8731928e-01


.. parsed-literal::

      11	-1.6683878e-02	 2.4285086e-01	-2.1990052e-03	 2.3617530e-01	  2.3698714e-02 	 2.0309019e-01


.. parsed-literal::

      12	-1.4372302e-02	 2.4246633e-01	-9.3963048e-05	 2.3567123e-01	[ 2.6939481e-02]	 2.0505190e-01


.. parsed-literal::

      13	-9.9943593e-03	 2.4162843e-01	 4.2273089e-03	 2.3466169e-01	[ 3.2463159e-02]	 2.1199512e-01


.. parsed-literal::

      14	 1.6740219e-01	 2.2796993e-01	 1.9175167e-01	 2.1973719e-01	[ 2.1295023e-01]	 4.2444658e-01


.. parsed-literal::

      15	 2.0607328e-01	 2.2272716e-01	 2.3103238e-01	 2.1474492e-01	[ 2.4577866e-01]	 2.9317451e-01
      16	 2.5474895e-01	 2.1802061e-01	 2.8191466e-01	 2.1124840e-01	[ 2.9487721e-01]	 1.7520833e-01


.. parsed-literal::

      17	 3.2522953e-01	 2.1274452e-01	 3.5714932e-01	 2.0644838e-01	[ 3.7446054e-01]	 2.0596862e-01


.. parsed-literal::

      18	 3.6724364e-01	 2.1146013e-01	 4.0011958e-01	 2.0554117e-01	[ 4.1170731e-01]	 2.0479369e-01


.. parsed-literal::

      19	 4.1942457e-01	 2.1126362e-01	 4.5228489e-01	 2.0686594e-01	[ 4.5870223e-01]	 2.0931935e-01


.. parsed-literal::

      20	 4.7887574e-01	 2.1013249e-01	 5.1258559e-01	 2.0656675e-01	[ 5.1026033e-01]	 2.0456195e-01


.. parsed-literal::

      21	 5.5628702e-01	 2.0900163e-01	 5.9212921e-01	 2.0429313e-01	[ 5.9112334e-01]	 2.0560122e-01


.. parsed-literal::

      22	 5.9370569e-01	 2.0831122e-01	 6.3251830e-01	 2.0204813e-01	[ 6.2843230e-01]	 2.0425844e-01
      23	 6.3086672e-01	 2.0435508e-01	 6.6860171e-01	 1.9849462e-01	[ 6.6860939e-01]	 1.8100762e-01


.. parsed-literal::

      24	 6.5413236e-01	 2.0268828e-01	 6.9149197e-01	 1.9644006e-01	[ 6.8935770e-01]	 2.1518350e-01


.. parsed-literal::

      25	 6.9346458e-01	 2.0023395e-01	 7.2938633e-01	 1.9362418e-01	[ 7.1834575e-01]	 3.2744002e-01


.. parsed-literal::

      26	 7.1604509e-01	 1.9987264e-01	 7.5176257e-01	 1.9270260e-01	[ 7.3227044e-01]	 3.1419992e-01
      27	 7.4815109e-01	 1.9844993e-01	 7.8523303e-01	 1.9089812e-01	[ 7.6295901e-01]	 2.0013833e-01


.. parsed-literal::

      28	 7.7165264e-01	 2.0150554e-01	 8.1158102e-01	 1.9334948e-01	[ 7.8266352e-01]	 2.0984459e-01
      29	 7.9739488e-01	 1.9954693e-01	 8.3622002e-01	 1.9432477e-01	[ 8.1078646e-01]	 2.0016241e-01


.. parsed-literal::

      30	 8.1513638e-01	 1.9678968e-01	 8.5422046e-01	 1.9112461e-01	[ 8.3319453e-01]	 1.9717336e-01
      31	 8.4324153e-01	 1.9378445e-01	 8.8359933e-01	 1.8598947e-01	[ 8.6710345e-01]	 1.7911410e-01


.. parsed-literal::

      32	 8.7017032e-01	 1.9331628e-01	 9.1113066e-01	 1.8573382e-01	[ 8.9439015e-01]	 1.7369318e-01
      33	 8.9684692e-01	 1.9445027e-01	 9.3831888e-01	 1.8739062e-01	[ 9.2172045e-01]	 2.0149398e-01


.. parsed-literal::

      34	 9.1213340e-01	 1.9541492e-01	 9.5359824e-01	 1.8805053e-01	[ 9.3816344e-01]	 2.0773673e-01


.. parsed-literal::

      35	 9.2617856e-01	 1.9296751e-01	 9.6762761e-01	 1.8603527e-01	[ 9.4968301e-01]	 2.1545529e-01


.. parsed-literal::

      36	 9.3806868e-01	 1.8998691e-01	 9.7967050e-01	 1.8216285e-01	[ 9.6190170e-01]	 2.0700788e-01


.. parsed-literal::

      37	 9.5249924e-01	 1.8755539e-01	 9.9436895e-01	 1.7952920e-01	[ 9.7406837e-01]	 2.0937991e-01


.. parsed-literal::

      38	 9.6833472e-01	 1.8412015e-01	 1.0107665e+00	 1.7559967e-01	[ 9.8881798e-01]	 2.1893620e-01


.. parsed-literal::

      39	 9.8276901e-01	 1.8250891e-01	 1.0260337e+00	 1.7304996e-01	[ 1.0030067e+00]	 2.0953178e-01


.. parsed-literal::

      40	 9.9665506e-01	 1.7988343e-01	 1.0405547e+00	 1.7016175e-01	[ 1.0170471e+00]	 2.1415544e-01


.. parsed-literal::

      41	 1.0101883e+00	 1.7841243e-01	 1.0544542e+00	 1.6884574e-01	[ 1.0293429e+00]	 2.0629811e-01


.. parsed-literal::

      42	 1.0289657e+00	 1.7690275e-01	 1.0753977e+00	 1.6801497e-01	[ 1.0493204e+00]	 2.0413232e-01
      43	 1.0384140e+00	 1.7572221e-01	 1.0844153e+00	 1.6663739e-01	[ 1.0536523e+00]	 1.6478133e-01


.. parsed-literal::

      44	 1.0465264e+00	 1.7456999e-01	 1.0920999e+00	 1.6575884e-01	[ 1.0623512e+00]	 2.0530891e-01


.. parsed-literal::

      45	 1.0537831e+00	 1.7267889e-01	 1.0994801e+00	 1.6431140e-01	[ 1.0692650e+00]	 2.0978069e-01
      46	 1.0611969e+00	 1.7093482e-01	 1.1069869e+00	 1.6256352e-01	[ 1.0741576e+00]	 1.7414784e-01


.. parsed-literal::

      47	 1.0733452e+00	 1.6649807e-01	 1.1195299e+00	 1.5792163e-01	[ 1.0815469e+00]	 1.7538238e-01
      48	 1.0825847e+00	 1.6503404e-01	 1.1289527e+00	 1.5559990e-01	[ 1.0888338e+00]	 1.7626643e-01


.. parsed-literal::

      49	 1.0901723e+00	 1.6507571e-01	 1.1362138e+00	 1.5525902e-01	[ 1.0968178e+00]	 2.0490623e-01


.. parsed-literal::

      50	 1.0993490e+00	 1.6357549e-01	 1.1454622e+00	 1.5366417e-01	[ 1.1072914e+00]	 2.1869874e-01


.. parsed-literal::

      51	 1.1092900e+00	 1.6096594e-01	 1.1555270e+00	 1.5142835e-01	[ 1.1141629e+00]	 2.1750450e-01
      52	 1.1160598e+00	 1.5894691e-01	 1.1625048e+00	 1.4928825e-01	[ 1.1238263e+00]	 1.8308163e-01


.. parsed-literal::

      53	 1.1229854e+00	 1.5852322e-01	 1.1692610e+00	 1.4836598e-01	[ 1.1317792e+00]	 2.0393872e-01


.. parsed-literal::

      54	 1.1337881e+00	 1.5716378e-01	 1.1803881e+00	 1.4623380e-01	[ 1.1416346e+00]	 2.0871615e-01
      55	 1.1421974e+00	 1.5603848e-01	 1.1892862e+00	 1.4429585e-01	[ 1.1530505e+00]	 1.7751384e-01


.. parsed-literal::

      56	 1.1518061e+00	 1.5444164e-01	 1.1995894e+00	 1.4264562e-01	[ 1.1639124e+00]	 2.1623802e-01
      57	 1.1601374e+00	 1.5348304e-01	 1.2080562e+00	 1.4155745e-01	[ 1.1745647e+00]	 1.9698167e-01


.. parsed-literal::

      58	 1.1682949e+00	 1.5307401e-01	 1.2163165e+00	 1.4158172e-01	[ 1.1823502e+00]	 1.9899869e-01


.. parsed-literal::

      59	 1.1749952e+00	 1.5196472e-01	 1.2235065e+00	 1.4115745e-01	[ 1.1886604e+00]	 2.0748353e-01


.. parsed-literal::

      60	 1.1822557e+00	 1.5141387e-01	 1.2307880e+00	 1.4071786e-01	[ 1.1943259e+00]	 2.1507049e-01


.. parsed-literal::

      61	 1.1904238e+00	 1.5078959e-01	 1.2391620e+00	 1.4026596e-01	[ 1.1990833e+00]	 2.0733500e-01


.. parsed-literal::

      62	 1.1978115e+00	 1.5034694e-01	 1.2466105e+00	 1.3992276e-01	[ 1.2052565e+00]	 2.1818733e-01
      63	 1.2050659e+00	 1.4998723e-01	 1.2540975e+00	 1.3983462e-01	[ 1.2079891e+00]	 1.9054794e-01


.. parsed-literal::

      64	 1.2115035e+00	 1.4953964e-01	 1.2604835e+00	 1.3933104e-01	[ 1.2146920e+00]	 1.8724394e-01


.. parsed-literal::

      65	 1.2201968e+00	 1.4862304e-01	 1.2693549e+00	 1.3838063e-01	[ 1.2244052e+00]	 2.0915937e-01
      66	 1.2293301e+00	 1.4766088e-01	 1.2786856e+00	 1.3770361e-01	[ 1.2320747e+00]	 2.0053291e-01


.. parsed-literal::

      67	 1.2344309e+00	 1.4679200e-01	 1.2844008e+00	 1.3726442e-01	[ 1.2324299e+00]	 2.1171522e-01


.. parsed-literal::

      68	 1.2430393e+00	 1.4629195e-01	 1.2926897e+00	 1.3670889e-01	[ 1.2415744e+00]	 2.1706605e-01


.. parsed-literal::

      69	 1.2462824e+00	 1.4609040e-01	 1.2959494e+00	 1.3659925e-01	  1.2414930e+00 	 2.1331739e-01


.. parsed-literal::

      70	 1.2538965e+00	 1.4584767e-01	 1.3037389e+00	 1.3634923e-01	[ 1.2443970e+00]	 2.0560074e-01


.. parsed-literal::

      71	 1.2611426e+00	 1.4504098e-01	 1.3115574e+00	 1.3605446e-01	  1.2372762e+00 	 2.0837474e-01
      72	 1.2683340e+00	 1.4506153e-01	 1.3185945e+00	 1.3583750e-01	[ 1.2452933e+00]	 1.9010854e-01


.. parsed-literal::

      73	 1.2732919e+00	 1.4470840e-01	 1.3234281e+00	 1.3537767e-01	[ 1.2555890e+00]	 1.8024158e-01
      74	 1.2792940e+00	 1.4410937e-01	 1.3296774e+00	 1.3480525e-01	[ 1.2593843e+00]	 1.8701148e-01


.. parsed-literal::

      75	 1.2865676e+00	 1.4305311e-01	 1.3372005e+00	 1.3390957e-01	[ 1.2615272e+00]	 2.1242809e-01
      76	 1.2916382e+00	 1.4224596e-01	 1.3423777e+00	 1.3296294e-01	  1.2592490e+00 	 1.9992471e-01


.. parsed-literal::

      77	 1.2970616e+00	 1.4195268e-01	 1.3477527e+00	 1.3270090e-01	  1.2605510e+00 	 2.0517707e-01


.. parsed-literal::

      78	 1.3052280e+00	 1.4150603e-01	 1.3560722e+00	 1.3244465e-01	  1.2604049e+00 	 2.1725702e-01


.. parsed-literal::

      79	 1.3090848e+00	 1.4072162e-01	 1.3602092e+00	 1.3221273e-01	  1.2594386e+00 	 2.0639706e-01
      80	 1.3151216e+00	 1.4053246e-01	 1.3661621e+00	 1.3212585e-01	[ 1.2634307e+00]	 1.8417740e-01


.. parsed-literal::

      81	 1.3199187e+00	 1.4024322e-01	 1.3710273e+00	 1.3207462e-01	[ 1.2640513e+00]	 2.0830488e-01


.. parsed-literal::

      82	 1.3252097e+00	 1.3971783e-01	 1.3765057e+00	 1.3178126e-01	  1.2630170e+00 	 2.0316029e-01
      83	 1.3342702e+00	 1.3881711e-01	 1.3858533e+00	 1.3105360e-01	  1.2612488e+00 	 2.0063949e-01


.. parsed-literal::

      84	 1.3400881e+00	 1.3824565e-01	 1.3920258e+00	 1.3016118e-01	  1.2622739e+00 	 3.1485844e-01


.. parsed-literal::

      85	 1.3471918e+00	 1.3774496e-01	 1.3993256e+00	 1.2940691e-01	  1.2615577e+00 	 2.0606208e-01
      86	 1.3523828e+00	 1.3735583e-01	 1.4044058e+00	 1.2876738e-01	[ 1.2691054e+00]	 1.7824626e-01


.. parsed-literal::

      87	 1.3570286e+00	 1.3713414e-01	 1.4091308e+00	 1.2808702e-01	[ 1.2713695e+00]	 2.1725845e-01
      88	 1.3613484e+00	 1.3655850e-01	 1.4134743e+00	 1.2727563e-01	[ 1.2785936e+00]	 2.0663381e-01


.. parsed-literal::

      89	 1.3670310e+00	 1.3594705e-01	 1.4190638e+00	 1.2652559e-01	[ 1.2853648e+00]	 1.9334769e-01


.. parsed-literal::

      90	 1.3731098e+00	 1.3489582e-01	 1.4252082e+00	 1.2534277e-01	[ 1.2903595e+00]	 2.1029878e-01


.. parsed-literal::

      91	 1.3785703e+00	 1.3438577e-01	 1.4306186e+00	 1.2517505e-01	[ 1.2936781e+00]	 2.1867585e-01


.. parsed-literal::

      92	 1.3825962e+00	 1.3423587e-01	 1.4344868e+00	 1.2501609e-01	[ 1.2984437e+00]	 2.0250821e-01


.. parsed-literal::

      93	 1.3879343e+00	 1.3375010e-01	 1.4400257e+00	 1.2461514e-01	[ 1.2990023e+00]	 2.1176219e-01
      94	 1.3923213e+00	 1.3346112e-01	 1.4445675e+00	 1.2410329e-01	  1.2962397e+00 	 1.8347621e-01


.. parsed-literal::

      95	 1.3968103e+00	 1.3309848e-01	 1.4491072e+00	 1.2365221e-01	  1.2963398e+00 	 2.0577192e-01


.. parsed-literal::

      96	 1.4021284e+00	 1.3239837e-01	 1.4546490e+00	 1.2282545e-01	  1.2945007e+00 	 2.0758390e-01
      97	 1.4055429e+00	 1.3218794e-01	 1.4583400e+00	 1.2252343e-01	  1.2888751e+00 	 1.8331838e-01


.. parsed-literal::

      98	 1.4095046e+00	 1.3205368e-01	 1.4621937e+00	 1.2232310e-01	  1.2939038e+00 	 1.9793010e-01


.. parsed-literal::

      99	 1.4127752e+00	 1.3185153e-01	 1.4655781e+00	 1.2211306e-01	  1.2917227e+00 	 2.0699430e-01
     100	 1.4157128e+00	 1.3165055e-01	 1.4684970e+00	 1.2180625e-01	  1.2937709e+00 	 1.8874192e-01


.. parsed-literal::

     101	 1.4215814e+00	 1.3110443e-01	 1.4744805e+00	 1.2101352e-01	  1.2935367e+00 	 1.8935418e-01


.. parsed-literal::

     102	 1.4240038e+00	 1.3079947e-01	 1.4772949e+00	 1.2012612e-01	  1.2885653e+00 	 2.1068072e-01


.. parsed-literal::

     103	 1.4289843e+00	 1.3058349e-01	 1.4819721e+00	 1.2018684e-01	  1.2984344e+00 	 2.0930505e-01


.. parsed-literal::

     104	 1.4315349e+00	 1.3044658e-01	 1.4845285e+00	 1.2024919e-01	[ 1.3004531e+00]	 2.0950007e-01


.. parsed-literal::

     105	 1.4357991e+00	 1.3019946e-01	 1.4888616e+00	 1.2021803e-01	[ 1.3048172e+00]	 2.0719576e-01


.. parsed-literal::

     106	 1.4383740e+00	 1.2998832e-01	 1.4914617e+00	 1.2033203e-01	[ 1.3068004e+00]	 3.2401299e-01


.. parsed-literal::

     107	 1.4421647e+00	 1.2984171e-01	 1.4953320e+00	 1.2014064e-01	[ 1.3097243e+00]	 2.0897150e-01


.. parsed-literal::

     108	 1.4460549e+00	 1.2960675e-01	 1.4992962e+00	 1.1987636e-01	[ 1.3115545e+00]	 2.0485425e-01
     109	 1.4493731e+00	 1.2943155e-01	 1.5027663e+00	 1.1984112e-01	  1.3076326e+00 	 1.8684483e-01


.. parsed-literal::

     110	 1.4527563e+00	 1.2923417e-01	 1.5062858e+00	 1.1999245e-01	  1.3003829e+00 	 2.0661163e-01


.. parsed-literal::

     111	 1.4559521e+00	 1.2896194e-01	 1.5095003e+00	 1.2001587e-01	  1.2946455e+00 	 2.1237469e-01
     112	 1.4594597e+00	 1.2880424e-01	 1.5130753e+00	 1.2017690e-01	  1.2886004e+00 	 1.8487906e-01


.. parsed-literal::

     113	 1.4628385e+00	 1.2871434e-01	 1.5165296e+00	 1.2027294e-01	  1.2852525e+00 	 2.1611714e-01


.. parsed-literal::

     114	 1.4665130e+00	 1.2859740e-01	 1.5203037e+00	 1.2019380e-01	  1.2898170e+00 	 2.2341847e-01


.. parsed-literal::

     115	 1.4703418e+00	 1.2851952e-01	 1.5241880e+00	 1.2024327e-01	  1.2867546e+00 	 2.1450353e-01


.. parsed-literal::

     116	 1.4724708e+00	 1.2844025e-01	 1.5263441e+00	 1.2033022e-01	  1.2861356e+00 	 2.1917439e-01


.. parsed-literal::

     117	 1.4749268e+00	 1.2834528e-01	 1.5288469e+00	 1.2041341e-01	  1.2810071e+00 	 2.0539498e-01


.. parsed-literal::

     118	 1.4774628e+00	 1.2820001e-01	 1.5315192e+00	 1.2055337e-01	  1.2727339e+00 	 2.1382570e-01


.. parsed-literal::

     119	 1.4795287e+00	 1.2807687e-01	 1.5335647e+00	 1.2047885e-01	  1.2716977e+00 	 2.1169710e-01
     120	 1.4821268e+00	 1.2783421e-01	 1.5361784e+00	 1.2031534e-01	  1.2686542e+00 	 1.9866204e-01


.. parsed-literal::

     121	 1.4839985e+00	 1.2768661e-01	 1.5381129e+00	 1.2023888e-01	  1.2715168e+00 	 2.0725536e-01
     122	 1.4861618e+00	 1.2759022e-01	 1.5402777e+00	 1.2017338e-01	  1.2748911e+00 	 1.9144464e-01


.. parsed-literal::

     123	 1.4886902e+00	 1.2742872e-01	 1.5428698e+00	 1.2011368e-01	  1.2772519e+00 	 2.1487808e-01


.. parsed-literal::

     124	 1.4905910e+00	 1.2731081e-01	 1.5448260e+00	 1.1999651e-01	  1.2777626e+00 	 2.1472692e-01


.. parsed-literal::

     125	 1.4937875e+00	 1.2694279e-01	 1.5481749e+00	 1.1986865e-01	  1.2722335e+00 	 2.0208573e-01
     126	 1.4960169e+00	 1.2682639e-01	 1.5503865e+00	 1.1985312e-01	  1.2701786e+00 	 1.7940235e-01


.. parsed-literal::

     127	 1.4977115e+00	 1.2670702e-01	 1.5520200e+00	 1.1985957e-01	  1.2721835e+00 	 1.7964053e-01


.. parsed-literal::

     128	 1.5002969e+00	 1.2644331e-01	 1.5546241e+00	 1.2003888e-01	  1.2699425e+00 	 2.0583200e-01
     129	 1.5018624e+00	 1.2619766e-01	 1.5562747e+00	 1.2021767e-01	  1.2729265e+00 	 1.9771981e-01


.. parsed-literal::

     130	 1.5036006e+00	 1.2611944e-01	 1.5580151e+00	 1.2027451e-01	  1.2713059e+00 	 2.1169329e-01


.. parsed-literal::

     131	 1.5057144e+00	 1.2593705e-01	 1.5602130e+00	 1.2030397e-01	  1.2684533e+00 	 2.1485186e-01


.. parsed-literal::

     132	 1.5071881e+00	 1.2579364e-01	 1.5617209e+00	 1.2022271e-01	  1.2654880e+00 	 2.1887517e-01
     133	 1.5093496e+00	 1.2561071e-01	 1.5638950e+00	 1.2006379e-01	  1.2654044e+00 	 1.8440890e-01


.. parsed-literal::

     134	 1.5111466e+00	 1.2544882e-01	 1.5656565e+00	 1.1996875e-01	  1.2678655e+00 	 2.0933342e-01


.. parsed-literal::

     135	 1.5128389e+00	 1.2536192e-01	 1.5672794e+00	 1.1990520e-01	  1.2699278e+00 	 2.0660806e-01


.. parsed-literal::

     136	 1.5144781e+00	 1.2531134e-01	 1.5688923e+00	 1.1992113e-01	  1.2710512e+00 	 2.0925236e-01


.. parsed-literal::

     137	 1.5160698e+00	 1.2512556e-01	 1.5705288e+00	 1.2003887e-01	  1.2649230e+00 	 2.0609450e-01


.. parsed-literal::

     138	 1.5175891e+00	 1.2505021e-01	 1.5721094e+00	 1.2004920e-01	  1.2610962e+00 	 2.1711397e-01


.. parsed-literal::

     139	 1.5187907e+00	 1.2496843e-01	 1.5733604e+00	 1.2006443e-01	  1.2583877e+00 	 2.0594764e-01


.. parsed-literal::

     140	 1.5204049e+00	 1.2484117e-01	 1.5750746e+00	 1.2009076e-01	  1.2545576e+00 	 2.0235801e-01


.. parsed-literal::

     141	 1.5214815e+00	 1.2478559e-01	 1.5762978e+00	 1.2006444e-01	  1.2563760e+00 	 2.0519209e-01
     142	 1.5227508e+00	 1.2476664e-01	 1.5775079e+00	 1.2003000e-01	  1.2567232e+00 	 1.9375324e-01


.. parsed-literal::

     143	 1.5240170e+00	 1.2474426e-01	 1.5787695e+00	 1.1995867e-01	  1.2573341e+00 	 1.9655895e-01


.. parsed-literal::

     144	 1.5252331e+00	 1.2472981e-01	 1.5799977e+00	 1.1988738e-01	  1.2546832e+00 	 2.0380402e-01
     145	 1.5269427e+00	 1.2460245e-01	 1.5818552e+00	 1.1962277e-01	  1.2456674e+00 	 1.8348312e-01


.. parsed-literal::

     146	 1.5287587e+00	 1.2462301e-01	 1.5836970e+00	 1.1954786e-01	  1.2375722e+00 	 2.1290708e-01


.. parsed-literal::

     147	 1.5295678e+00	 1.2453865e-01	 1.5845136e+00	 1.1950889e-01	  1.2370755e+00 	 2.1341729e-01


.. parsed-literal::

     148	 1.5308901e+00	 1.2434389e-01	 1.5859276e+00	 1.1934412e-01	  1.2334294e+00 	 2.1956682e-01
     149	 1.5323700e+00	 1.2410454e-01	 1.5875127e+00	 1.1916107e-01	  1.2282136e+00 	 2.0136189e-01


.. parsed-literal::

     150	 1.5328375e+00	 1.2375313e-01	 1.5883154e+00	 1.1884378e-01	  1.2163274e+00 	 1.8587971e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.04 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7faef02d2f50>



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
    CPU times: user 1.74 s, sys: 39 ms, total: 1.78 s
    Wall time: 552 ms


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

