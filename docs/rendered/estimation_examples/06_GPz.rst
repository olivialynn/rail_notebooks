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
       1	-3.4474916e-01	 3.2121069e-01	-3.3516417e-01	 3.1837109e-01	[-3.3002078e-01]	 4.6624851e-01


.. parsed-literal::

       2	-2.7736189e-01	 3.1194500e-01	-2.5483396e-01	 3.0823256e-01	[-2.4380389e-01]	 2.3093867e-01


.. parsed-literal::

       3	-2.3107847e-01	 2.8945707e-01	-1.8835993e-01	 2.8324725e-01	[-1.6522343e-01]	 2.8559732e-01
       4	-1.9060461e-01	 2.6657592e-01	-1.4919957e-01	 2.6395746e-01	[-1.3188724e-01]	 1.7301035e-01


.. parsed-literal::

       5	-9.8144327e-02	 2.5674718e-01	-6.6532996e-02	 2.5247974e-01	[-4.8224224e-02]	 2.1248078e-01


.. parsed-literal::

       6	-7.1309990e-02	 2.5256527e-01	-4.3353103e-02	 2.5230659e-01	[-3.6701383e-02]	 2.1314454e-01


.. parsed-literal::

       7	-5.2802136e-02	 2.4933276e-01	-3.0331009e-02	 2.4724008e-01	[-1.9305456e-02]	 2.0257044e-01


.. parsed-literal::

       8	-4.0951286e-02	 2.4732049e-01	-2.1828821e-02	 2.4487937e-01	[-1.0139535e-02]	 2.1043658e-01


.. parsed-literal::

       9	-2.8912471e-02	 2.4503188e-01	-1.2031495e-02	 2.4260915e-01	[-8.8890434e-04]	 2.1326280e-01


.. parsed-literal::

      10	-1.9594156e-02	 2.4294240e-01	-4.4621893e-03	 2.3982320e-01	[ 1.0709659e-02]	 2.0804644e-01
      11	-1.3422926e-02	 2.4209618e-01	 6.9520986e-04	 2.3934899e-01	[ 1.2474973e-02]	 1.8886399e-01


.. parsed-literal::

      12	-1.0543933e-02	 2.4164759e-01	 3.5111398e-03	 2.3887502e-01	[ 1.5764589e-02]	 2.0481896e-01
      13	-6.6498876e-03	 2.4079001e-01	 7.6184236e-03	 2.3817703e-01	[ 1.9154310e-02]	 1.7785048e-01


.. parsed-literal::

      14	 1.3245594e-01	 2.2632786e-01	 1.5802890e-01	 2.2433293e-01	[ 1.5233584e-01]	 4.3292761e-01


.. parsed-literal::

      15	 2.0153965e-01	 2.2714437e-01	 2.2870176e-01	 2.2847419e-01	[ 2.2090506e-01]	 3.3100295e-01
      16	 2.6193456e-01	 2.2008804e-01	 2.9108264e-01	 2.1959176e-01	[ 2.8030349e-01]	 1.9785738e-01


.. parsed-literal::

      17	 3.1537237e-01	 2.1766899e-01	 3.4682821e-01	 2.1654728e-01	[ 3.2816337e-01]	 2.0934868e-01


.. parsed-literal::

      18	 3.7921857e-01	 2.1248971e-01	 4.1280830e-01	 2.1228249e-01	[ 3.9079907e-01]	 2.2060513e-01


.. parsed-literal::

      19	 4.2606892e-01	 2.1002703e-01	 4.5925119e-01	 2.1102204e-01	[ 4.3546786e-01]	 2.1023345e-01


.. parsed-literal::

      20	 4.6704687e-01	 2.0935173e-01	 5.0157376e-01	 2.1153214e-01	[ 4.7344290e-01]	 2.2463298e-01


.. parsed-literal::

      21	 5.1495699e-01	 2.0677113e-01	 5.5130706e-01	 2.0916030e-01	[ 5.1839644e-01]	 2.0366502e-01


.. parsed-literal::

      22	 5.5985363e-01	 2.0425777e-01	 5.9853866e-01	 2.0611178e-01	[ 5.6884635e-01]	 2.0993567e-01


.. parsed-literal::

      23	 5.9328247e-01	 2.0272042e-01	 6.3168555e-01	 2.0613189e-01	[ 5.9738879e-01]	 2.1864676e-01


.. parsed-literal::

      24	 6.2647623e-01	 2.0067236e-01	 6.6473183e-01	 2.0356427e-01	[ 6.2793838e-01]	 2.1095729e-01


.. parsed-literal::

      25	 6.7933375e-01	 2.0164707e-01	 7.1685839e-01	 2.0299432e-01	[ 6.7809060e-01]	 2.0163369e-01


.. parsed-literal::

      26	 6.8067657e-01	 2.1263131e-01	 7.1625631e-01	 2.1327392e-01	[ 6.8409677e-01]	 2.2023797e-01


.. parsed-literal::

      27	 7.5494813e-01	 2.0591345e-01	 7.9406632e-01	 2.0541274e-01	[ 7.4180395e-01]	 2.2210836e-01


.. parsed-literal::

      28	 7.7048158e-01	 2.0529073e-01	 8.1088961e-01	 2.0381459e-01	[ 7.6477410e-01]	 2.1046329e-01


.. parsed-literal::

      29	 7.9132816e-01	 2.0509800e-01	 8.3086613e-01	 2.0396992e-01	[ 7.8439393e-01]	 2.1852636e-01


.. parsed-literal::

      30	 8.1393432e-01	 2.0558775e-01	 8.5330703e-01	 2.0509788e-01	[ 8.0582592e-01]	 2.0890427e-01
      31	 8.3446653e-01	 2.0777969e-01	 8.7443079e-01	 2.0723142e-01	[ 8.2850059e-01]	 1.9987249e-01


.. parsed-literal::

      32	 8.5570494e-01	 2.1276798e-01	 8.9713469e-01	 2.1336943e-01	[ 8.4728784e-01]	 2.1047449e-01


.. parsed-literal::

      33	 8.7163275e-01	 2.1167212e-01	 9.1380616e-01	 2.1128111e-01	[ 8.6628227e-01]	 2.1180105e-01


.. parsed-literal::

      34	 8.8776400e-01	 2.0916016e-01	 9.2963249e-01	 2.0950012e-01	[ 8.8013922e-01]	 2.1114039e-01


.. parsed-literal::

      35	 9.1480658e-01	 2.0529135e-01	 9.5715078e-01	 2.0680659e-01	[ 9.0437776e-01]	 2.0825434e-01


.. parsed-literal::

      36	 9.3393110e-01	 2.0227259e-01	 9.7725718e-01	 2.0360035e-01	[ 9.3154109e-01]	 2.0537138e-01


.. parsed-literal::

      37	 9.4960075e-01	 1.9805646e-01	 9.9441885e-01	 1.9992542e-01	[ 9.4553937e-01]	 2.0494056e-01


.. parsed-literal::

      38	 9.6153790e-01	 1.9548495e-01	 1.0068734e+00	 1.9768318e-01	[ 9.5580207e-01]	 2.1103573e-01


.. parsed-literal::

      39	 9.7765679e-01	 1.9339287e-01	 1.0239884e+00	 1.9655378e-01	[ 9.6357110e-01]	 2.0634961e-01


.. parsed-literal::

      40	 9.9114585e-01	 1.9210201e-01	 1.0373219e+00	 1.9550779e-01	[ 9.7285331e-01]	 2.0641112e-01
      41	 1.0031912e+00	 1.9076374e-01	 1.0496452e+00	 1.9505267e-01	[ 9.7980076e-01]	 1.7576623e-01


.. parsed-literal::

      42	 1.0117309e+00	 1.9013705e-01	 1.0577961e+00	 1.9467339e-01	[ 9.8559314e-01]	 1.8180680e-01


.. parsed-literal::

      43	 1.0309387e+00	 1.8410190e-01	 1.0772193e+00	 1.8991552e-01	[ 9.8941413e-01]	 2.0863247e-01
      44	 1.0423764e+00	 1.8173238e-01	 1.0891867e+00	 1.8817988e-01	[ 9.9254799e-01]	 1.9524550e-01


.. parsed-literal::

      45	 1.0536007e+00	 1.7910076e-01	 1.1001829e+00	 1.8606877e-01	[ 9.9939681e-01]	 2.0523953e-01


.. parsed-literal::

      46	 1.0635329e+00	 1.7619753e-01	 1.1102251e+00	 1.8380984e-01	[ 1.0021674e+00]	 2.2257686e-01


.. parsed-literal::

      47	 1.0741979e+00	 1.7425220e-01	 1.1212478e+00	 1.8234641e-01	[ 1.0052618e+00]	 2.1392179e-01


.. parsed-literal::

      48	 1.0900393e+00	 1.7305339e-01	 1.1372530e+00	 1.8161616e-01	[ 1.0270213e+00]	 2.1876574e-01


.. parsed-literal::

      49	 1.1023148e+00	 1.7179706e-01	 1.1497816e+00	 1.8055981e-01	[ 1.0398630e+00]	 2.1304679e-01
      50	 1.1100137e+00	 1.7050836e-01	 1.1577913e+00	 1.7983896e-01	[ 1.0472361e+00]	 1.9943619e-01


.. parsed-literal::

      51	 1.1309249e+00	 1.6312767e-01	 1.1797724e+00	 1.7470527e-01	[ 1.0744403e+00]	 1.7216945e-01


.. parsed-literal::

      52	 1.1438212e+00	 1.5921514e-01	 1.1934281e+00	 1.7085965e-01	[ 1.0892107e+00]	 2.0605755e-01


.. parsed-literal::

      53	 1.1584729e+00	 1.5420330e-01	 1.2079763e+00	 1.6698191e-01	[ 1.1087433e+00]	 2.0623374e-01


.. parsed-literal::

      54	 1.1668524e+00	 1.5481080e-01	 1.2159886e+00	 1.6663925e-01	[ 1.1141800e+00]	 2.0144820e-01
      55	 1.1775905e+00	 1.5267769e-01	 1.2268208e+00	 1.6469222e-01	[ 1.1255878e+00]	 2.0862031e-01


.. parsed-literal::

      56	 1.1935942e+00	 1.4919657e-01	 1.2432636e+00	 1.6247900e-01	[ 1.1400225e+00]	 1.7433262e-01


.. parsed-literal::

      57	 1.2057099e+00	 1.4948734e-01	 1.2551935e+00	 1.6264309e-01	[ 1.1513795e+00]	 2.0381474e-01


.. parsed-literal::

      58	 1.2198324e+00	 1.4713390e-01	 1.2693172e+00	 1.6126499e-01	[ 1.1591502e+00]	 2.1692729e-01


.. parsed-literal::

      59	 1.2275133e+00	 1.4632020e-01	 1.2771492e+00	 1.6067431e-01	[ 1.1646473e+00]	 2.1196818e-01


.. parsed-literal::

      60	 1.2389631e+00	 1.4540077e-01	 1.2891752e+00	 1.5994978e-01	[ 1.1735999e+00]	 2.0954800e-01
      61	 1.2525723e+00	 1.4509057e-01	 1.3031935e+00	 1.5912141e-01	[ 1.1867361e+00]	 2.0544815e-01


.. parsed-literal::

      62	 1.2627395e+00	 1.4435602e-01	 1.3140285e+00	 1.5724026e-01	[ 1.1999295e+00]	 1.8901706e-01


.. parsed-literal::

      63	 1.2768495e+00	 1.4321621e-01	 1.3277084e+00	 1.5573071e-01	[ 1.2147534e+00]	 2.2069931e-01


.. parsed-literal::

      64	 1.2879291e+00	 1.4197645e-01	 1.3388535e+00	 1.5360418e-01	[ 1.2234655e+00]	 2.1061683e-01


.. parsed-literal::

      65	 1.2974798e+00	 1.4108606e-01	 1.3484456e+00	 1.5188499e-01	[ 1.2349814e+00]	 2.2103357e-01
      66	 1.3081430e+00	 1.4077852e-01	 1.3591023e+00	 1.4952995e-01	[ 1.2434067e+00]	 1.9981122e-01


.. parsed-literal::

      67	 1.3181806e+00	 1.3951398e-01	 1.3693416e+00	 1.4811226e-01	[ 1.2487120e+00]	 2.1366477e-01


.. parsed-literal::

      68	 1.3279200e+00	 1.3914293e-01	 1.3795035e+00	 1.4602084e-01	[ 1.2573920e+00]	 2.1026111e-01


.. parsed-literal::

      69	 1.3366276e+00	 1.3692133e-01	 1.3880234e+00	 1.4418163e-01	[ 1.2617098e+00]	 2.0861673e-01


.. parsed-literal::

      70	 1.3416875e+00	 1.3636659e-01	 1.3930261e+00	 1.4364156e-01	[ 1.2667095e+00]	 2.1264839e-01


.. parsed-literal::

      71	 1.3527317e+00	 1.3507216e-01	 1.4045252e+00	 1.4147286e-01	[ 1.2706883e+00]	 2.1012044e-01
      72	 1.3580481e+00	 1.3443201e-01	 1.4097286e+00	 1.4067162e-01	[ 1.2721172e+00]	 1.8798637e-01


.. parsed-literal::

      73	 1.3652383e+00	 1.3405234e-01	 1.4168819e+00	 1.3963069e-01	[ 1.2797827e+00]	 2.1252608e-01
      74	 1.3710674e+00	 1.3362613e-01	 1.4229877e+00	 1.3848599e-01	  1.2797391e+00 	 1.9799328e-01


.. parsed-literal::

      75	 1.3765768e+00	 1.3313142e-01	 1.4287384e+00	 1.3766340e-01	[ 1.2800196e+00]	 1.7491055e-01


.. parsed-literal::

      76	 1.3856160e+00	 1.3225502e-01	 1.4384044e+00	 1.3707566e-01	[ 1.2808752e+00]	 2.0520353e-01


.. parsed-literal::

      77	 1.3874374e+00	 1.3151663e-01	 1.4406996e+00	 1.3649574e-01	  1.2761776e+00 	 2.1893311e-01


.. parsed-literal::

      78	 1.3977587e+00	 1.3145255e-01	 1.4503911e+00	 1.3711734e-01	[ 1.2912116e+00]	 2.1323895e-01


.. parsed-literal::

      79	 1.4014071e+00	 1.3136440e-01	 1.4540923e+00	 1.3731506e-01	[ 1.2912757e+00]	 2.2056127e-01
      80	 1.4070237e+00	 1.3123822e-01	 1.4597818e+00	 1.3732667e-01	[ 1.2915848e+00]	 2.0397615e-01


.. parsed-literal::

      81	 1.4126668e+00	 1.3103812e-01	 1.4655922e+00	 1.3824457e-01	  1.2828035e+00 	 1.9130731e-01


.. parsed-literal::

      82	 1.4190504e+00	 1.3080837e-01	 1.4719818e+00	 1.3719757e-01	  1.2831636e+00 	 2.1316099e-01


.. parsed-literal::

      83	 1.4241326e+00	 1.3068812e-01	 1.4771089e+00	 1.3703087e-01	  1.2810500e+00 	 2.0751691e-01


.. parsed-literal::

      84	 1.4298932e+00	 1.3067853e-01	 1.4829205e+00	 1.3688841e-01	  1.2731431e+00 	 2.1922469e-01
      85	 1.4320345e+00	 1.3120075e-01	 1.4852929e+00	 1.3818021e-01	  1.2594257e+00 	 1.9465137e-01


.. parsed-literal::

      86	 1.4372376e+00	 1.3051596e-01	 1.4902662e+00	 1.3718775e-01	  1.2688100e+00 	 1.8981934e-01


.. parsed-literal::

      87	 1.4403763e+00	 1.3027109e-01	 1.4933956e+00	 1.3681575e-01	  1.2694206e+00 	 2.0982838e-01


.. parsed-literal::

      88	 1.4450334e+00	 1.2985752e-01	 1.4981489e+00	 1.3612219e-01	  1.2692848e+00 	 2.1110129e-01


.. parsed-literal::

      89	 1.4517399e+00	 1.2950425e-01	 1.5050489e+00	 1.3559583e-01	  1.2640259e+00 	 2.1212196e-01


.. parsed-literal::

      90	 1.4564848e+00	 1.2939842e-01	 1.5099761e+00	 1.3540911e-01	  1.2568314e+00 	 3.2777882e-01


.. parsed-literal::

      91	 1.4608252e+00	 1.2903702e-01	 1.5144134e+00	 1.3468936e-01	  1.2550772e+00 	 2.1222591e-01


.. parsed-literal::

      92	 1.4646777e+00	 1.2871763e-01	 1.5183525e+00	 1.3418836e-01	  1.2486724e+00 	 2.1349359e-01


.. parsed-literal::

      93	 1.4683970e+00	 1.2829337e-01	 1.5221312e+00	 1.3364327e-01	  1.2400887e+00 	 2.0842600e-01
      94	 1.4722014e+00	 1.2787271e-01	 1.5258793e+00	 1.3289139e-01	  1.2386224e+00 	 1.7951536e-01


.. parsed-literal::

      95	 1.4755686e+00	 1.2769031e-01	 1.5292169e+00	 1.3276420e-01	  1.2325723e+00 	 2.1076465e-01
      96	 1.4785814e+00	 1.2749594e-01	 1.5322029e+00	 1.3254776e-01	  1.2308831e+00 	 1.9723773e-01


.. parsed-literal::

      97	 1.4818715e+00	 1.2732258e-01	 1.5355235e+00	 1.3230524e-01	  1.2274757e+00 	 2.0883632e-01


.. parsed-literal::

      98	 1.4849104e+00	 1.2739398e-01	 1.5385774e+00	 1.3276038e-01	  1.2206699e+00 	 2.0533872e-01
      99	 1.4875432e+00	 1.2728529e-01	 1.5412287e+00	 1.3242769e-01	  1.2151811e+00 	 2.0073318e-01


.. parsed-literal::

     100	 1.4926417e+00	 1.2717829e-01	 1.5464546e+00	 1.3170024e-01	  1.1865427e+00 	 2.0568013e-01


.. parsed-literal::

     101	 1.4940009e+00	 1.2681306e-01	 1.5479492e+00	 1.3001998e-01	  1.1619838e+00 	 2.0293331e-01
     102	 1.4978201e+00	 1.2683491e-01	 1.5516143e+00	 1.3050769e-01	  1.1679943e+00 	 2.0025849e-01


.. parsed-literal::

     103	 1.4997362e+00	 1.2683574e-01	 1.5535550e+00	 1.3072585e-01	  1.1586289e+00 	 1.8845034e-01


.. parsed-literal::

     104	 1.5030238e+00	 1.2678092e-01	 1.5569611e+00	 1.3088703e-01	  1.1335427e+00 	 2.0617175e-01
     105	 1.5061264e+00	 1.2678684e-01	 1.5602876e+00	 1.3146092e-01	  1.0970763e+00 	 1.9514751e-01


.. parsed-literal::

     106	 1.5089706e+00	 1.2665493e-01	 1.5632846e+00	 1.3102084e-01	  1.0650217e+00 	 2.0976782e-01
     107	 1.5109169e+00	 1.2648791e-01	 1.5651556e+00	 1.3066050e-01	  1.0738953e+00 	 1.8993068e-01


.. parsed-literal::

     108	 1.5128114e+00	 1.2639280e-01	 1.5670964e+00	 1.3040748e-01	  1.0689118e+00 	 2.2069860e-01


.. parsed-literal::

     109	 1.5147760e+00	 1.2611895e-01	 1.5691779e+00	 1.2976159e-01	  1.0602118e+00 	 2.0576286e-01


.. parsed-literal::

     110	 1.5172895e+00	 1.2601821e-01	 1.5717046e+00	 1.2950840e-01	  1.0491468e+00 	 2.1284771e-01
     111	 1.5201702e+00	 1.2598511e-01	 1.5746472e+00	 1.2952456e-01	  1.0267035e+00 	 2.0596027e-01


.. parsed-literal::

     112	 1.5218249e+00	 1.2584782e-01	 1.5763054e+00	 1.2928808e-01	  1.0178896e+00 	 2.0238709e-01


.. parsed-literal::

     113	 1.5241063e+00	 1.2574558e-01	 1.5785759e+00	 1.2938803e-01	  1.0091325e+00 	 2.1655369e-01


.. parsed-literal::

     114	 1.5265656e+00	 1.2548008e-01	 1.5811354e+00	 1.2918445e-01	  9.9701482e-01 	 2.0998669e-01
     115	 1.5287600e+00	 1.2547302e-01	 1.5833341e+00	 1.2947577e-01	  9.8472181e-01 	 1.9939327e-01


.. parsed-literal::

     116	 1.5308911e+00	 1.2541597e-01	 1.5854993e+00	 1.2949642e-01	  9.7408037e-01 	 2.0525098e-01


.. parsed-literal::

     117	 1.5348666e+00	 1.2514567e-01	 1.5896593e+00	 1.2952428e-01	  9.3173955e-01 	 2.1075392e-01


.. parsed-literal::

     118	 1.5364402e+00	 1.2500414e-01	 1.5912807e+00	 1.2958992e-01	  9.1510966e-01 	 3.2625961e-01


.. parsed-literal::

     119	 1.5379252e+00	 1.2488703e-01	 1.5927593e+00	 1.2956304e-01	  9.0349268e-01 	 2.0670867e-01


.. parsed-literal::

     120	 1.5403583e+00	 1.2459328e-01	 1.5952316e+00	 1.2964762e-01	  8.6995431e-01 	 2.1664333e-01


.. parsed-literal::

     121	 1.5418315e+00	 1.2441822e-01	 1.5967745e+00	 1.2964919e-01	  8.4731720e-01 	 2.1068335e-01


.. parsed-literal::

     122	 1.5434558e+00	 1.2430791e-01	 1.5984177e+00	 1.2969543e-01	  8.3380862e-01 	 2.1041274e-01


.. parsed-literal::

     123	 1.5465387e+00	 1.2406335e-01	 1.6016647e+00	 1.2967691e-01	  7.9905565e-01 	 2.0983243e-01


.. parsed-literal::

     124	 1.5480135e+00	 1.2402774e-01	 1.6031816e+00	 1.2990902e-01	  7.9640815e-01 	 2.0960784e-01
     125	 1.5496762e+00	 1.2387485e-01	 1.6048342e+00	 1.2967862e-01	  7.9718980e-01 	 1.8275452e-01


.. parsed-literal::

     126	 1.5512866e+00	 1.2375609e-01	 1.6064300e+00	 1.2958977e-01	  7.9084052e-01 	 2.0803070e-01


.. parsed-literal::

     127	 1.5526518e+00	 1.2359159e-01	 1.6077849e+00	 1.2947520e-01	  7.8797744e-01 	 2.1254134e-01


.. parsed-literal::

     128	 1.5548452e+00	 1.2343012e-01	 1.6099958e+00	 1.2982727e-01	  7.6915626e-01 	 2.1312881e-01
     129	 1.5563254e+00	 1.2322598e-01	 1.6115287e+00	 1.2977039e-01	  7.5063308e-01 	 1.9769001e-01


.. parsed-literal::

     130	 1.5576847e+00	 1.2318403e-01	 1.6128719e+00	 1.2991087e-01	  7.4898914e-01 	 2.0953059e-01


.. parsed-literal::

     131	 1.5593509e+00	 1.2309304e-01	 1.6146048e+00	 1.3012178e-01	  7.4009753e-01 	 2.1117687e-01


.. parsed-literal::

     132	 1.5608148e+00	 1.2302043e-01	 1.6161237e+00	 1.3021933e-01	  7.2889095e-01 	 2.2086048e-01


.. parsed-literal::

     133	 1.5616661e+00	 1.2280068e-01	 1.6172716e+00	 1.3051421e-01	  6.9654245e-01 	 2.1001291e-01


.. parsed-literal::

     134	 1.5642126e+00	 1.2280760e-01	 1.6196662e+00	 1.3036696e-01	  6.8759506e-01 	 2.1275806e-01


.. parsed-literal::

     135	 1.5650211e+00	 1.2280415e-01	 1.6204413e+00	 1.3033619e-01	  6.8220886e-01 	 2.1970201e-01


.. parsed-literal::

     136	 1.5667404e+00	 1.2272643e-01	 1.6221780e+00	 1.3032670e-01	  6.6206487e-01 	 2.0252657e-01


.. parsed-literal::

     137	 1.5676672e+00	 1.2282247e-01	 1.6232190e+00	 1.3068905e-01	  6.4046290e-01 	 2.1519494e-01


.. parsed-literal::

     138	 1.5692430e+00	 1.2267435e-01	 1.6247201e+00	 1.3052383e-01	  6.4156849e-01 	 2.1072483e-01


.. parsed-literal::

     139	 1.5702288e+00	 1.2257461e-01	 1.6257228e+00	 1.3052730e-01	  6.3809612e-01 	 2.1942186e-01


.. parsed-literal::

     140	 1.5713482e+00	 1.2250231e-01	 1.6268644e+00	 1.3053726e-01	  6.3106945e-01 	 2.0651245e-01
     141	 1.5726090e+00	 1.2228749e-01	 1.6282470e+00	 1.3039734e-01	  6.0402089e-01 	 2.0775342e-01


.. parsed-literal::

     142	 1.5739187e+00	 1.2229561e-01	 1.6295488e+00	 1.3041671e-01	  5.8211747e-01 	 2.1421790e-01
     143	 1.5748311e+00	 1.2227977e-01	 1.6304327e+00	 1.3038776e-01	  5.7320415e-01 	 1.9957876e-01


.. parsed-literal::

     144	 1.5764365e+00	 1.2219387e-01	 1.6320591e+00	 1.3043558e-01	  5.3454335e-01 	 1.9941449e-01
     145	 1.5775579e+00	 1.2211702e-01	 1.6332151e+00	 1.3049393e-01	  4.9537886e-01 	 1.9512630e-01


.. parsed-literal::

     146	 1.5790316e+00	 1.2197615e-01	 1.6347009e+00	 1.3061008e-01	  4.4770404e-01 	 2.1063042e-01


.. parsed-literal::

     147	 1.5806891e+00	 1.2173814e-01	 1.6363613e+00	 1.3071262e-01	  3.9781289e-01 	 2.1339869e-01


.. parsed-literal::

     148	 1.5818339e+00	 1.2151018e-01	 1.6375129e+00	 1.3064437e-01	  3.4274934e-01 	 2.1095824e-01


.. parsed-literal::

     149	 1.5827724e+00	 1.2149511e-01	 1.6384153e+00	 1.3061705e-01	  3.4452753e-01 	 2.0820117e-01


.. parsed-literal::

     150	 1.5837492e+00	 1.2141010e-01	 1.6393755e+00	 1.3039810e-01	  3.3571798e-01 	 2.0504594e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.12 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1134a70d90>



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
    CPU times: user 2.19 s, sys: 44 ms, total: 2.24 s
    Wall time: 702 ms


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

