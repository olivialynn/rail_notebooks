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
       1	-3.4065902e-01	 3.1999266e-01	-3.3087884e-01	 3.2463696e-01	[-3.3736517e-01]	 4.6004081e-01


.. parsed-literal::

       2	-2.7103986e-01	 3.0941929e-01	-2.4664717e-01	 3.1157988e-01	[-2.5332005e-01]	 2.3021555e-01


.. parsed-literal::

       3	-2.2330640e-01	 2.8658510e-01	-1.7815455e-01	 2.9451865e-01	[-2.0506108e-01]	 2.8787541e-01


.. parsed-literal::

       4	-1.8594045e-01	 2.6959541e-01	-1.3609178e-01	 2.8086262e-01	[-1.8356630e-01]	 3.2017732e-01
       5	-1.1812214e-01	 2.5712983e-01	-8.4843582e-02	 2.6507830e-01	[-1.2612945e-01]	 1.7632055e-01


.. parsed-literal::

       6	-6.9997597e-02	 2.5199396e-01	-4.1066307e-02	 2.5936059e-01	[-6.9443523e-02]	 2.0784235e-01


.. parsed-literal::

       7	-5.0072808e-02	 2.4805861e-01	-2.5576101e-02	 2.5483510e-01	[-5.2652431e-02]	 2.1756387e-01


.. parsed-literal::

       8	-3.5260527e-02	 2.4555509e-01	-1.5010622e-02	 2.5233450e-01	[-4.4641213e-02]	 2.1538162e-01
       9	-2.1431903e-02	 2.4267458e-01	-3.9357213e-03	 2.5182745e-01	[-3.9406019e-02]	 1.7943668e-01


.. parsed-literal::

      10	-8.6381567e-03	 2.4060575e-01	 6.7304488e-03	 2.5050116e-01	[-3.6200385e-02]	 2.1060157e-01
      11	-3.3596123e-03	 2.3946767e-01	 1.1697660e-02	 2.4993678e-01	[-3.4027506e-02]	 1.7269802e-01


.. parsed-literal::

      12	 3.1733893e-03	 2.3838253e-01	 1.7208626e-02	 2.4970494e-01	[-3.2377541e-02]	 2.0311022e-01


.. parsed-literal::

      13	 7.3975935e-03	 2.3775049e-01	 2.1008697e-02	 2.5058505e-01	 -3.2792483e-02 	 2.1696401e-01


.. parsed-literal::

      14	 7.0504521e-02	 2.2567016e-01	 9.0557831e-02	 2.3426444e-01	[ 6.6717398e-02]	 4.3921566e-01
      15	 1.3016437e-01	 2.2129087e-01	 1.5339305e-01	 2.3145077e-01	[ 1.2287917e-01]	 1.7714453e-01


.. parsed-literal::

      16	 2.3047379e-01	 2.1863704e-01	 2.6078171e-01	 2.6332765e-01	[ 1.8411888e-01]	 2.2335672e-01


.. parsed-literal::

      17	 2.9946675e-01	 2.1605375e-01	 3.2938679e-01	 2.3253256e-01	[ 2.7903487e-01]	 2.1495223e-01


.. parsed-literal::

      18	 3.4680305e-01	 2.1039287e-01	 3.7757777e-01	 2.2944743e-01	[ 3.3074622e-01]	 2.1308279e-01


.. parsed-literal::

      19	 4.1007255e-01	 2.0645478e-01	 4.4265177e-01	 2.2162413e-01	[ 4.0451612e-01]	 2.0816398e-01


.. parsed-literal::

      20	 4.7743362e-01	 2.0266645e-01	 5.1168608e-01	 2.1822433e-01	[ 4.7289898e-01]	 2.1412754e-01


.. parsed-literal::

      21	 5.3716615e-01	 1.9891119e-01	 5.7420396e-01	 2.1449062e-01	[ 5.3333822e-01]	 2.1558857e-01
      22	 5.9764513e-01	 1.9345008e-01	 6.3705968e-01	 2.0854641e-01	[ 6.0585034e-01]	 1.8315148e-01


.. parsed-literal::

      23	 6.3247379e-01	 1.9179434e-01	 6.7233855e-01	 2.0288963e-01	[ 6.6287417e-01]	 2.0944238e-01


.. parsed-literal::

      24	 6.7394478e-01	 1.8484519e-01	 7.1415737e-01	 1.9543074e-01	[ 7.0579546e-01]	 2.1169043e-01
      25	 7.0137144e-01	 1.8317155e-01	 7.4085875e-01	 1.9377917e-01	[ 7.2483118e-01]	 1.9830632e-01


.. parsed-literal::

      26	 7.4281947e-01	 1.8162405e-01	 7.8167171e-01	 1.9120464e-01	[ 7.6421914e-01]	 2.1275330e-01


.. parsed-literal::

      27	 7.7047350e-01	 1.9359691e-01	 8.0869162e-01	 2.0175238e-01	[ 7.9016420e-01]	 2.1214652e-01


.. parsed-literal::

      28	 8.1253443e-01	 1.8567204e-01	 8.5163744e-01	 1.9309274e-01	[ 8.3301635e-01]	 2.1385932e-01


.. parsed-literal::

      29	 8.4501116e-01	 1.8430096e-01	 8.8556729e-01	 1.9294372e-01	[ 8.6597507e-01]	 2.1021438e-01


.. parsed-literal::

      30	 8.7510292e-01	 1.8100460e-01	 9.1604911e-01	 1.8643497e-01	[ 9.0139088e-01]	 2.1864486e-01


.. parsed-literal::

      31	 8.9836898e-01	 1.7921460e-01	 9.3941606e-01	 1.8218296e-01	[ 9.2771680e-01]	 2.1308684e-01


.. parsed-literal::

      32	 9.1421088e-01	 1.7545044e-01	 9.5576434e-01	 1.8057771e-01	[ 9.4726842e-01]	 2.1724939e-01
      33	 9.2885313e-01	 1.7437714e-01	 9.7060025e-01	 1.7943811e-01	[ 9.6134357e-01]	 1.8078160e-01


.. parsed-literal::

      34	 9.5544629e-01	 1.7073532e-01	 9.9878533e-01	 1.7563175e-01	[ 9.9206244e-01]	 2.0938730e-01
      35	 9.7221756e-01	 1.6913189e-01	 1.0155867e+00	 1.7513103e-01	[ 9.9909969e-01]	 1.9575906e-01


.. parsed-literal::

      36	 9.8870595e-01	 1.6726730e-01	 1.0321441e+00	 1.7289607e-01	[ 1.0166440e+00]	 2.1783233e-01
      37	 1.0061593e+00	 1.6418726e-01	 1.0502254e+00	 1.6918005e-01	[ 1.0248993e+00]	 1.9907546e-01


.. parsed-literal::

      38	 1.0159058e+00	 1.6251920e-01	 1.0607896e+00	 1.6687937e-01	[ 1.0313499e+00]	 2.0952654e-01
      39	 1.0268505e+00	 1.6099076e-01	 1.0719202e+00	 1.6453179e-01	[ 1.0421492e+00]	 2.0158195e-01


.. parsed-literal::

      40	 1.0412657e+00	 1.5929862e-01	 1.0868763e+00	 1.6228831e-01	[ 1.0533643e+00]	 2.0974398e-01


.. parsed-literal::

      41	 1.0543911e+00	 1.5809570e-01	 1.1006581e+00	 1.6032766e-01	[ 1.0675545e+00]	 2.1099901e-01
      42	 1.0666330e+00	 1.5650401e-01	 1.1132505e+00	 1.5916589e-01	[ 1.0771217e+00]	 1.8554091e-01


.. parsed-literal::

      43	 1.0753232e+00	 1.5501253e-01	 1.1221091e+00	 1.5734268e-01	[ 1.0844559e+00]	 2.1570420e-01


.. parsed-literal::

      44	 1.0870201e+00	 1.5313542e-01	 1.1339602e+00	 1.5557010e-01	[ 1.0940987e+00]	 2.1440196e-01


.. parsed-literal::

      45	 1.0973102e+00	 1.5156965e-01	 1.1445018e+00	 1.5384477e-01	[ 1.1074134e+00]	 2.1966314e-01


.. parsed-literal::

      46	 1.1086196e+00	 1.5054770e-01	 1.1555993e+00	 1.5354504e-01	[ 1.1150210e+00]	 2.1145058e-01


.. parsed-literal::

      47	 1.1193303e+00	 1.4963822e-01	 1.1661611e+00	 1.5323479e-01	[ 1.1248982e+00]	 2.0727015e-01
      48	 1.1335195e+00	 1.4818730e-01	 1.1806409e+00	 1.5210701e-01	[ 1.1376215e+00]	 1.9460392e-01


.. parsed-literal::

      49	 1.1410380e+00	 1.4753722e-01	 1.1883094e+00	 1.5082702e-01	[ 1.1507739e+00]	 2.0818949e-01


.. parsed-literal::

      50	 1.1522448e+00	 1.4640304e-01	 1.1993981e+00	 1.4996761e-01	[ 1.1601899e+00]	 2.1913838e-01
      51	 1.1611704e+00	 1.4525874e-01	 1.2083688e+00	 1.4923204e-01	[ 1.1665810e+00]	 1.8425369e-01


.. parsed-literal::

      52	 1.1697807e+00	 1.4399254e-01	 1.2171835e+00	 1.4806349e-01	[ 1.1734096e+00]	 1.8381023e-01


.. parsed-literal::

      53	 1.1843537e+00	 1.4209041e-01	 1.2321503e+00	 1.4550938e-01	[ 1.1794902e+00]	 2.0150113e-01


.. parsed-literal::

      54	 1.1864078e+00	 1.4145490e-01	 1.2353332e+00	 1.4375221e-01	[ 1.1846674e+00]	 2.1755147e-01


.. parsed-literal::

      55	 1.2008630e+00	 1.4087053e-01	 1.2492116e+00	 1.4324916e-01	[ 1.1896948e+00]	 2.0827293e-01
      56	 1.2057415e+00	 1.4083293e-01	 1.2540768e+00	 1.4314235e-01	[ 1.1903281e+00]	 1.8879676e-01


.. parsed-literal::

      57	 1.2148958e+00	 1.4053630e-01	 1.2634531e+00	 1.4257312e-01	[ 1.1926006e+00]	 1.9740295e-01


.. parsed-literal::

      58	 1.2254184e+00	 1.4055042e-01	 1.2741445e+00	 1.4247899e-01	[ 1.1970784e+00]	 2.1091270e-01


.. parsed-literal::

      59	 1.2328966e+00	 1.4008980e-01	 1.2820727e+00	 1.4221927e-01	[ 1.1990676e+00]	 2.1389413e-01


.. parsed-literal::

      60	 1.2409866e+00	 1.3973907e-01	 1.2899341e+00	 1.4215216e-01	[ 1.2065451e+00]	 2.2106886e-01


.. parsed-literal::

      61	 1.2483625e+00	 1.3936261e-01	 1.2973541e+00	 1.4210492e-01	[ 1.2123665e+00]	 2.0894694e-01


.. parsed-literal::

      62	 1.2573916e+00	 1.3918344e-01	 1.3065657e+00	 1.4204608e-01	[ 1.2193906e+00]	 2.1017981e-01


.. parsed-literal::

      63	 1.2648616e+00	 1.3895771e-01	 1.3142618e+00	 1.4221709e-01	[ 1.2243763e+00]	 3.2411766e-01


.. parsed-literal::

      64	 1.2740437e+00	 1.3867149e-01	 1.3236245e+00	 1.4176438e-01	[ 1.2332034e+00]	 2.0898366e-01


.. parsed-literal::

      65	 1.2827705e+00	 1.3850136e-01	 1.3325958e+00	 1.4136736e-01	[ 1.2396895e+00]	 2.1301866e-01
      66	 1.2891049e+00	 1.3815206e-01	 1.3392950e+00	 1.4046902e-01	[ 1.2483021e+00]	 1.9581938e-01


.. parsed-literal::

      67	 1.2944699e+00	 1.3785169e-01	 1.3446944e+00	 1.4004572e-01	[ 1.2516995e+00]	 2.1019745e-01


.. parsed-literal::

      68	 1.2995884e+00	 1.3787214e-01	 1.3497455e+00	 1.3978025e-01	[ 1.2559207e+00]	 2.1091485e-01


.. parsed-literal::

      69	 1.3082226e+00	 1.3716999e-01	 1.3585264e+00	 1.3844758e-01	[ 1.2642243e+00]	 2.1977544e-01


.. parsed-literal::

      70	 1.3158599e+00	 1.3666688e-01	 1.3664367e+00	 1.3698320e-01	  1.2585721e+00 	 2.0822001e-01


.. parsed-literal::

      71	 1.3233641e+00	 1.3575665e-01	 1.3738068e+00	 1.3581007e-01	[ 1.2658394e+00]	 2.0724225e-01


.. parsed-literal::

      72	 1.3304795e+00	 1.3464881e-01	 1.3809909e+00	 1.3423991e-01	[ 1.2668920e+00]	 2.0800424e-01


.. parsed-literal::

      73	 1.3377681e+00	 1.3396312e-01	 1.3884793e+00	 1.3324995e-01	  1.2639186e+00 	 2.1251655e-01


.. parsed-literal::

      74	 1.3407972e+00	 1.3328423e-01	 1.3923001e+00	 1.3158172e-01	  1.2442667e+00 	 2.2355580e-01


.. parsed-literal::

      75	 1.3502539e+00	 1.3312534e-01	 1.4014547e+00	 1.3197473e-01	  1.2540789e+00 	 2.1183658e-01


.. parsed-literal::

      76	 1.3529327e+00	 1.3291680e-01	 1.4041116e+00	 1.3161571e-01	  1.2596076e+00 	 2.1350980e-01


.. parsed-literal::

      77	 1.3592264e+00	 1.3275061e-01	 1.4105692e+00	 1.3091629e-01	  1.2629962e+00 	 2.0494485e-01


.. parsed-literal::

      78	 1.3655242e+00	 1.3216053e-01	 1.4171514e+00	 1.2989293e-01	  1.2552013e+00 	 2.1205020e-01


.. parsed-literal::

      79	 1.3717798e+00	 1.3222507e-01	 1.4233330e+00	 1.2955418e-01	  1.2616514e+00 	 2.0944333e-01


.. parsed-literal::

      80	 1.3773374e+00	 1.3234283e-01	 1.4288877e+00	 1.2923412e-01	  1.2600446e+00 	 2.0481157e-01


.. parsed-literal::

      81	 1.3834044e+00	 1.3230991e-01	 1.4350037e+00	 1.2832736e-01	  1.2630500e+00 	 2.1116042e-01
      82	 1.3876412e+00	 1.3252939e-01	 1.4393594e+00	 1.2814263e-01	  1.2638513e+00 	 2.0291877e-01


.. parsed-literal::

      83	 1.3908608e+00	 1.3219584e-01	 1.4424784e+00	 1.2784615e-01	[ 1.2703617e+00]	 2.0978284e-01


.. parsed-literal::

      84	 1.3946350e+00	 1.3193282e-01	 1.4463166e+00	 1.2747010e-01	[ 1.2727781e+00]	 2.0765376e-01
      85	 1.3991917e+00	 1.3172355e-01	 1.4509825e+00	 1.2685213e-01	[ 1.2756182e+00]	 1.9932222e-01


.. parsed-literal::

      86	 1.4036953e+00	 1.3173127e-01	 1.4556187e+00	 1.2657099e-01	  1.2725084e+00 	 2.1661258e-01


.. parsed-literal::

      87	 1.4085020e+00	 1.3180242e-01	 1.4604932e+00	 1.2600615e-01	  1.2734528e+00 	 2.0828462e-01


.. parsed-literal::

      88	 1.4120488e+00	 1.3176778e-01	 1.4639743e+00	 1.2583410e-01	[ 1.2789963e+00]	 2.0830488e-01
      89	 1.4162210e+00	 1.3170019e-01	 1.4682272e+00	 1.2527273e-01	[ 1.2796187e+00]	 1.9550395e-01


.. parsed-literal::

      90	 1.4190549e+00	 1.3179614e-01	 1.4713317e+00	 1.2501285e-01	[ 1.2812805e+00]	 2.1706676e-01
      91	 1.4221906e+00	 1.3151015e-01	 1.4744375e+00	 1.2488151e-01	  1.2796247e+00 	 1.9355369e-01


.. parsed-literal::

      92	 1.4242357e+00	 1.3134732e-01	 1.4765671e+00	 1.2471567e-01	  1.2766757e+00 	 2.2236443e-01
      93	 1.4265277e+00	 1.3116191e-01	 1.4788515e+00	 1.2452239e-01	  1.2785290e+00 	 1.9909310e-01


.. parsed-literal::

      94	 1.4299714e+00	 1.3102964e-01	 1.4823424e+00	 1.2426646e-01	  1.2763304e+00 	 1.8297505e-01
      95	 1.4334269e+00	 1.3075808e-01	 1.4856474e+00	 1.2395141e-01	[ 1.2873595e+00]	 1.9869733e-01


.. parsed-literal::

      96	 1.4357578e+00	 1.3063869e-01	 1.4878753e+00	 1.2401071e-01	[ 1.2911450e+00]	 2.0819807e-01


.. parsed-literal::

      97	 1.4387748e+00	 1.3050613e-01	 1.4909151e+00	 1.2401387e-01	[ 1.2938162e+00]	 2.0785069e-01
      98	 1.4417570e+00	 1.3041810e-01	 1.4940609e+00	 1.2434782e-01	  1.2931984e+00 	 1.6972208e-01


.. parsed-literal::

      99	 1.4447306e+00	 1.3045288e-01	 1.4971309e+00	 1.2451631e-01	[ 1.2947044e+00]	 1.8764114e-01


.. parsed-literal::

     100	 1.4474151e+00	 1.3046587e-01	 1.4999290e+00	 1.2465461e-01	[ 1.2949536e+00]	 2.1422076e-01
     101	 1.4497007e+00	 1.3057302e-01	 1.5022221e+00	 1.2463700e-01	[ 1.2972281e+00]	 1.8358254e-01


.. parsed-literal::

     102	 1.4523461e+00	 1.3061217e-01	 1.5048322e+00	 1.2464128e-01	[ 1.2990108e+00]	 2.1276975e-01


.. parsed-literal::

     103	 1.4545281e+00	 1.3082543e-01	 1.5070826e+00	 1.2503720e-01	[ 1.2997570e+00]	 3.2527399e-01


.. parsed-literal::

     104	 1.4575343e+00	 1.3084316e-01	 1.5100918e+00	 1.2512027e-01	  1.2965184e+00 	 2.0250726e-01


.. parsed-literal::

     105	 1.4600015e+00	 1.3096515e-01	 1.5126060e+00	 1.2509084e-01	  1.2966433e+00 	 2.1458435e-01
     106	 1.4633775e+00	 1.3109017e-01	 1.5161086e+00	 1.2492683e-01	  1.2946551e+00 	 1.8674850e-01


.. parsed-literal::

     107	 1.4656204e+00	 1.3106046e-01	 1.5185067e+00	 1.2461383e-01	  1.2951904e+00 	 2.1120501e-01


.. parsed-literal::

     108	 1.4680066e+00	 1.3100810e-01	 1.5208681e+00	 1.2429553e-01	[ 1.3034777e+00]	 2.2169518e-01


.. parsed-literal::

     109	 1.4699007e+00	 1.3095830e-01	 1.5227557e+00	 1.2431999e-01	[ 1.3048905e+00]	 2.1027255e-01
     110	 1.4723673e+00	 1.3082399e-01	 1.5252120e+00	 1.2418715e-01	[ 1.3062326e+00]	 1.8808579e-01


.. parsed-literal::

     111	 1.4753093e+00	 1.2997396e-01	 1.5281600e+00	 1.2394325e-01	  1.3049720e+00 	 2.0520473e-01


.. parsed-literal::

     112	 1.4777336e+00	 1.2998423e-01	 1.5305475e+00	 1.2393815e-01	  1.3045580e+00 	 2.1033812e-01


.. parsed-literal::

     113	 1.4791855e+00	 1.2978116e-01	 1.5319435e+00	 1.2369383e-01	  1.3045325e+00 	 2.1217370e-01


.. parsed-literal::

     114	 1.4810797e+00	 1.2935442e-01	 1.5338565e+00	 1.2354934e-01	  1.3029619e+00 	 2.0264626e-01


.. parsed-literal::

     115	 1.4830272e+00	 1.2898478e-01	 1.5358610e+00	 1.2334484e-01	  1.2990818e+00 	 2.1929383e-01


.. parsed-literal::

     116	 1.4854483e+00	 1.2861480e-01	 1.5383529e+00	 1.2313793e-01	  1.2977562e+00 	 2.1756339e-01
     117	 1.4896850e+00	 1.2796395e-01	 1.5427103e+00	 1.2260458e-01	  1.2956159e+00 	 1.8513680e-01


.. parsed-literal::

     118	 1.4910056e+00	 1.2783880e-01	 1.5440882e+00	 1.2238370e-01	  1.2934963e+00 	 3.3610749e-01


.. parsed-literal::

     119	 1.4926779e+00	 1.2798467e-01	 1.5457714e+00	 1.2239150e-01	  1.2930706e+00 	 2.0889020e-01


.. parsed-literal::

     120	 1.4941633e+00	 1.2796391e-01	 1.5472560e+00	 1.2230122e-01	  1.2904816e+00 	 2.1171522e-01
     121	 1.4955881e+00	 1.2801962e-01	 1.5487409e+00	 1.2228607e-01	  1.2852093e+00 	 1.9856191e-01


.. parsed-literal::

     122	 1.4974036e+00	 1.2787775e-01	 1.5505572e+00	 1.2211285e-01	  1.2803396e+00 	 2.0769691e-01


.. parsed-literal::

     123	 1.4987025e+00	 1.2769105e-01	 1.5519010e+00	 1.2192948e-01	  1.2761453e+00 	 2.1667480e-01
     124	 1.5005138e+00	 1.2750411e-01	 1.5538120e+00	 1.2174183e-01	  1.2683633e+00 	 1.9968247e-01


.. parsed-literal::

     125	 1.5023404e+00	 1.2740230e-01	 1.5558325e+00	 1.2143927e-01	  1.2556782e+00 	 2.1935105e-01


.. parsed-literal::

     126	 1.5041510e+00	 1.2735554e-01	 1.5576934e+00	 1.2145244e-01	  1.2497982e+00 	 2.1528602e-01


.. parsed-literal::

     127	 1.5055014e+00	 1.2735815e-01	 1.5590426e+00	 1.2145690e-01	  1.2481208e+00 	 2.1306205e-01


.. parsed-literal::

     128	 1.5073820e+00	 1.2748673e-01	 1.5609351e+00	 1.2133218e-01	  1.2457093e+00 	 2.0269275e-01


.. parsed-literal::

     129	 1.5085210e+00	 1.2728704e-01	 1.5620819e+00	 1.2132949e-01	  1.2413045e+00 	 2.1896982e-01


.. parsed-literal::

     130	 1.5099066e+00	 1.2734248e-01	 1.5634132e+00	 1.2121013e-01	  1.2450006e+00 	 2.1474838e-01


.. parsed-literal::

     131	 1.5111522e+00	 1.2735661e-01	 1.5646345e+00	 1.2108400e-01	  1.2442029e+00 	 2.1576810e-01


.. parsed-literal::

     132	 1.5126670e+00	 1.2743087e-01	 1.5661365e+00	 1.2114571e-01	  1.2446133e+00 	 2.1803284e-01


.. parsed-literal::

     133	 1.5143253e+00	 1.2755490e-01	 1.5678519e+00	 1.2108110e-01	  1.2397015e+00 	 2.0537114e-01


.. parsed-literal::

     134	 1.5158971e+00	 1.2756040e-01	 1.5694249e+00	 1.2127116e-01	  1.2418823e+00 	 2.1296406e-01
     135	 1.5171873e+00	 1.2754952e-01	 1.5707595e+00	 1.2144434e-01	  1.2432760e+00 	 1.9927859e-01


.. parsed-literal::

     136	 1.5183858e+00	 1.2757074e-01	 1.5720016e+00	 1.2156264e-01	  1.2466969e+00 	 2.1551514e-01


.. parsed-literal::

     137	 1.5190039e+00	 1.2742076e-01	 1.5727934e+00	 1.2163021e-01	  1.2469127e+00 	 2.1916127e-01


.. parsed-literal::

     138	 1.5206562e+00	 1.2749143e-01	 1.5743556e+00	 1.2159220e-01	  1.2505860e+00 	 2.0757318e-01


.. parsed-literal::

     139	 1.5212968e+00	 1.2747251e-01	 1.5749646e+00	 1.2153591e-01	  1.2509121e+00 	 2.1491385e-01


.. parsed-literal::

     140	 1.5226652e+00	 1.2738652e-01	 1.5763136e+00	 1.2151766e-01	  1.2498175e+00 	 2.1954179e-01


.. parsed-literal::

     141	 1.5247988e+00	 1.2720986e-01	 1.5784577e+00	 1.2151722e-01	  1.2496018e+00 	 2.2022796e-01


.. parsed-literal::

     142	 1.5258961e+00	 1.2705328e-01	 1.5795971e+00	 1.2168352e-01	  1.2487187e+00 	 3.3099866e-01


.. parsed-literal::

     143	 1.5273739e+00	 1.2696962e-01	 1.5810954e+00	 1.2176252e-01	  1.2509370e+00 	 2.0458436e-01


.. parsed-literal::

     144	 1.5286276e+00	 1.2689983e-01	 1.5823505e+00	 1.2181901e-01	  1.2536702e+00 	 2.0573950e-01


.. parsed-literal::

     145	 1.5299829e+00	 1.2688849e-01	 1.5837352e+00	 1.2195790e-01	  1.2590393e+00 	 2.0935822e-01


.. parsed-literal::

     146	 1.5315022e+00	 1.2672727e-01	 1.5852427e+00	 1.2194738e-01	  1.2603341e+00 	 2.0481777e-01


.. parsed-literal::

     147	 1.5329475e+00	 1.2660484e-01	 1.5866746e+00	 1.2190414e-01	  1.2603379e+00 	 2.0387816e-01


.. parsed-literal::

     148	 1.5345921e+00	 1.2639444e-01	 1.5883598e+00	 1.2183399e-01	  1.2556765e+00 	 2.0685005e-01


.. parsed-literal::

     149	 1.5358240e+00	 1.2627181e-01	 1.5896671e+00	 1.2202274e-01	  1.2504579e+00 	 2.0829153e-01


.. parsed-literal::

     150	 1.5371172e+00	 1.2611272e-01	 1.5910252e+00	 1.2208591e-01	  1.2440093e+00 	 2.0251822e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 8s, sys: 1.14 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc0ec700f70>



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
    CPU times: user 1.79 s, sys: 46 ms, total: 1.83 s
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

