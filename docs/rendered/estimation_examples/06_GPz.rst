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
       1	-3.5397270e-01	 3.2395329e-01	-3.4432012e-01	 3.0716772e-01	[-3.1358702e-01]	 4.5947123e-01


.. parsed-literal::

       2	-2.8359306e-01	 3.1339519e-01	-2.5937429e-01	 2.9700005e-01	[-2.1078174e-01]	 2.3136377e-01


.. parsed-literal::

       3	-2.3782001e-01	 2.9121914e-01	-1.9401630e-01	 2.7769688e-01	[-1.3688645e-01]	 2.8508568e-01
       4	-2.0103768e-01	 2.6801856e-01	-1.5959982e-01	 2.5775319e-01	[-9.8810258e-02]	 1.9966483e-01


.. parsed-literal::

       5	-1.0492770e-01	 2.5766904e-01	-7.0854377e-02	 2.4952280e-01	[-3.3441973e-02]	 2.0568275e-01
       6	-7.7253417e-02	 2.5399249e-01	-4.8090041e-02	 2.4793299e-01	[-2.1027754e-02]	 1.9958735e-01


.. parsed-literal::

       7	-5.7757296e-02	 2.5038753e-01	-3.4482965e-02	 2.4361428e-01	[-6.2669109e-03]	 1.7699885e-01
       8	-4.7107847e-02	 2.4863862e-01	-2.7233809e-02	 2.4132352e-01	[ 2.8731543e-03]	 1.9547248e-01


.. parsed-literal::

       9	-3.5762198e-02	 2.4656217e-01	-1.8372337e-02	 2.3853428e-01	[ 1.4139236e-02]	 1.6761446e-01


.. parsed-literal::

      10	-2.4322911e-02	 2.4409776e-01	-8.5245504e-03	 2.3591207e-01	[ 2.7401227e-02]	 2.0472789e-01


.. parsed-literal::

      11	-1.8762558e-02	 2.4314872e-01	-3.8818835e-03	 2.3516773e-01	[ 2.7705927e-02]	 3.1606650e-01


.. parsed-literal::

      12	-1.4989686e-02	 2.4257337e-01	-5.3278972e-04	 2.3474379e-01	[ 3.1780953e-02]	 2.0750046e-01


.. parsed-literal::

      13	-1.0175455e-02	 2.4164714e-01	 4.1322355e-03	 2.3402739e-01	[ 3.6182225e-02]	 2.1898556e-01


.. parsed-literal::

      14	 1.7950434e-01	 2.2415353e-01	 2.0472938e-01	 2.1575937e-01	[ 2.2786372e-01]	 4.1348863e-01


.. parsed-literal::

      15	 1.9505829e-01	 2.2616895e-01	 2.2151013e-01	 2.2000939e-01	[ 2.3326707e-01]	 2.0956564e-01


.. parsed-literal::

      16	 2.9253298e-01	 2.2072774e-01	 3.2182807e-01	 2.1354352e-01	[ 3.4208153e-01]	 2.1808910e-01


.. parsed-literal::

      17	 3.5568828e-01	 2.1762176e-01	 3.8943647e-01	 2.1230254e-01	[ 4.1638232e-01]	 2.0287251e-01


.. parsed-literal::

      18	 4.1772036e-01	 2.1299901e-01	 4.5194427e-01	 2.0878757e-01	[ 4.7079025e-01]	 2.0194411e-01
      19	 4.6588042e-01	 2.0957381e-01	 5.0037666e-01	 2.0517622e-01	[ 5.1959595e-01]	 1.9638586e-01


.. parsed-literal::

      20	 5.6451540e-01	 2.0335934e-01	 6.0107983e-01	 2.0004423e-01	[ 6.3062502e-01]	 2.0145226e-01
      21	 5.9892971e-01	 2.0671434e-01	 6.4039067e-01	 2.0360395e-01	[ 6.9299773e-01]	 1.9855738e-01


.. parsed-literal::

      22	 6.5702343e-01	 1.9864648e-01	 6.9525957e-01	 1.9630779e-01	[ 7.2146013e-01]	 2.1804643e-01
      23	 6.8355231e-01	 1.9697174e-01	 7.2200076e-01	 1.9493025e-01	[ 7.4793112e-01]	 1.8144345e-01


.. parsed-literal::

      24	 7.1958027e-01	 2.0236846e-01	 7.5738201e-01	 1.9812037e-01	[ 7.7788418e-01]	 2.1641755e-01
      25	 7.6720250e-01	 1.9762647e-01	 8.0578563e-01	 1.9423411e-01	[ 8.1300349e-01]	 1.7294788e-01


.. parsed-literal::

      26	 7.9362449e-01	 1.9467810e-01	 8.3362280e-01	 1.9123698e-01	[ 8.3073583e-01]	 2.1235943e-01


.. parsed-literal::

      27	 8.2633390e-01	 1.9372422e-01	 8.6704981e-01	 1.8955488e-01	[ 8.5710674e-01]	 2.0602942e-01
      28	 8.5631685e-01	 1.9110843e-01	 8.9845380e-01	 1.8330306e-01	[ 8.7275295e-01]	 1.7643118e-01


.. parsed-literal::

      29	 8.8423674e-01	 1.8989797e-01	 9.2684406e-01	 1.8287143e-01	[ 9.0010969e-01]	 2.0429778e-01
      30	 9.0272908e-01	 1.8892865e-01	 9.4544047e-01	 1.8264642e-01	[ 9.1675465e-01]	 2.0145822e-01


.. parsed-literal::

      31	 9.2636124e-01	 1.8516901e-01	 9.6992930e-01	 1.7770857e-01	[ 9.4134755e-01]	 2.0565367e-01


.. parsed-literal::

      32	 9.4823721e-01	 1.8116645e-01	 9.9167695e-01	 1.7486876e-01	[ 9.6365631e-01]	 2.1251082e-01


.. parsed-literal::

      33	 9.7124612e-01	 1.7706471e-01	 1.0147519e+00	 1.7102555e-01	[ 9.8674337e-01]	 2.0662999e-01
      34	 9.8977242e-01	 1.7365372e-01	 1.0337063e+00	 1.6866646e-01	[ 1.0017860e+00]	 1.8530059e-01


.. parsed-literal::

      35	 1.0074253e+00	 1.7015358e-01	 1.0523452e+00	 1.6588307e-01	[ 1.0212252e+00]	 2.0479155e-01
      36	 1.0249691e+00	 1.6685732e-01	 1.0708709e+00	 1.6278543e-01	[ 1.0319981e+00]	 1.9963694e-01


.. parsed-literal::

      37	 1.0452362e+00	 1.6212611e-01	 1.0923112e+00	 1.5737586e-01	[ 1.0484674e+00]	 1.7735243e-01


.. parsed-literal::

      38	 1.0606228e+00	 1.5801512e-01	 1.1080297e+00	 1.5356676e-01	[ 1.0621772e+00]	 2.0571637e-01


.. parsed-literal::

      39	 1.0762394e+00	 1.5511016e-01	 1.1238366e+00	 1.5115101e-01	[ 1.0771452e+00]	 2.1214032e-01
      40	 1.0895408e+00	 1.5356195e-01	 1.1373957e+00	 1.5050588e-01	[ 1.0884642e+00]	 1.7608237e-01


.. parsed-literal::

      41	 1.1044353e+00	 1.5240034e-01	 1.1520763e+00	 1.5089594e-01	[ 1.1027810e+00]	 2.0909166e-01


.. parsed-literal::

      42	 1.1182449e+00	 1.5087716e-01	 1.1657603e+00	 1.5151054e-01	[ 1.1155421e+00]	 2.0492816e-01


.. parsed-literal::

      43	 1.1346326e+00	 1.4924845e-01	 1.1820003e+00	 1.4882852e-01	[ 1.1384946e+00]	 2.2093964e-01


.. parsed-literal::

      44	 1.1384964e+00	 1.4906769e-01	 1.1862390e+00	 1.5328229e-01	[ 1.1468854e+00]	 2.1450043e-01


.. parsed-literal::

      45	 1.1530084e+00	 1.4661368e-01	 1.2003175e+00	 1.4709285e-01	[ 1.1626786e+00]	 2.1077156e-01
      46	 1.1586613e+00	 1.4680007e-01	 1.2061874e+00	 1.4592297e-01	[ 1.1692205e+00]	 1.9229746e-01


.. parsed-literal::

      47	 1.1672477e+00	 1.4577885e-01	 1.2150478e+00	 1.4407963e-01	[ 1.1794549e+00]	 2.1642113e-01


.. parsed-literal::

      48	 1.1896863e+00	 1.4281231e-01	 1.2384241e+00	 1.4031772e-01	[ 1.2012252e+00]	 2.0312667e-01


.. parsed-literal::

      49	 1.2024835e+00	 1.4137803e-01	 1.2515446e+00	 1.3716185e-01	[ 1.2203506e+00]	 2.0748854e-01


.. parsed-literal::

      50	 1.2163158e+00	 1.4002186e-01	 1.2652977e+00	 1.3707049e-01	[ 1.2276028e+00]	 2.1413136e-01
      51	 1.2280174e+00	 1.3904837e-01	 1.2770645e+00	 1.3710132e-01	[ 1.2344289e+00]	 1.7947483e-01


.. parsed-literal::

      52	 1.2375806e+00	 1.3779653e-01	 1.2866566e+00	 1.3724444e-01	[ 1.2412698e+00]	 2.0850086e-01


.. parsed-literal::

      53	 1.2505098e+00	 1.3619537e-01	 1.2997944e+00	 1.3635701e-01	[ 1.2500721e+00]	 2.0696282e-01


.. parsed-literal::

      54	 1.2623595e+00	 1.3527417e-01	 1.3118961e+00	 1.3531558e-01	[ 1.2629883e+00]	 2.0753074e-01
      55	 1.2736826e+00	 1.3485419e-01	 1.3234620e+00	 1.3422483e-01	[ 1.2673498e+00]	 1.9861984e-01


.. parsed-literal::

      56	 1.2843203e+00	 1.3472814e-01	 1.3343557e+00	 1.3358535e-01	[ 1.2765626e+00]	 2.0067000e-01
      57	 1.2955634e+00	 1.3382261e-01	 1.3459861e+00	 1.3452600e-01	[ 1.2778599e+00]	 1.6768479e-01


.. parsed-literal::

      58	 1.3039876e+00	 1.3413179e-01	 1.3546215e+00	 1.3525941e-01	  1.2769502e+00 	 1.9884372e-01


.. parsed-literal::

      59	 1.3121893e+00	 1.3331199e-01	 1.3627072e+00	 1.3533311e-01	[ 1.2824737e+00]	 2.0874739e-01


.. parsed-literal::

      60	 1.3219925e+00	 1.3228373e-01	 1.3726098e+00	 1.3604173e-01	[ 1.2874846e+00]	 2.0427442e-01
      61	 1.3321505e+00	 1.3146471e-01	 1.3825710e+00	 1.3631394e-01	[ 1.3014132e+00]	 1.7126560e-01


.. parsed-literal::

      62	 1.3401937e+00	 1.3120886e-01	 1.3908530e+00	 1.3843628e-01	  1.3011487e+00 	 2.0608258e-01
      63	 1.3483244e+00	 1.3052250e-01	 1.3990080e+00	 1.3766667e-01	[ 1.3119139e+00]	 1.8740392e-01


.. parsed-literal::

      64	 1.3561878e+00	 1.2980388e-01	 1.4071903e+00	 1.3643049e-01	[ 1.3177864e+00]	 2.0866203e-01


.. parsed-literal::

      65	 1.3633460e+00	 1.2915291e-01	 1.4144912e+00	 1.3621861e-01	[ 1.3230430e+00]	 2.1199775e-01
      66	 1.3726235e+00	 1.2845040e-01	 1.4241777e+00	 1.3586407e-01	  1.3172201e+00 	 1.9620466e-01


.. parsed-literal::

      67	 1.3789188e+00	 1.2802267e-01	 1.4305794e+00	 1.3598437e-01	[ 1.3237619e+00]	 1.7419839e-01


.. parsed-literal::

      68	 1.3847260e+00	 1.2775359e-01	 1.4364380e+00	 1.3585981e-01	[ 1.3265981e+00]	 2.1892905e-01


.. parsed-literal::

      69	 1.3906826e+00	 1.2759422e-01	 1.4424130e+00	 1.3513637e-01	[ 1.3319433e+00]	 2.1659136e-01


.. parsed-literal::

      70	 1.3945271e+00	 1.2799465e-01	 1.4465792e+00	 1.3639157e-01	[ 1.3336140e+00]	 2.1098638e-01
      71	 1.4001284e+00	 1.2750578e-01	 1.4519826e+00	 1.3530174e-01	[ 1.3397109e+00]	 2.0219493e-01


.. parsed-literal::

      72	 1.4037440e+00	 1.2730077e-01	 1.4555777e+00	 1.3521249e-01	[ 1.3411720e+00]	 2.1783209e-01
      73	 1.4085372e+00	 1.2702273e-01	 1.4603066e+00	 1.3527365e-01	[ 1.3417347e+00]	 1.7470121e-01


.. parsed-literal::

      74	 1.4150616e+00	 1.2666899e-01	 1.4669045e+00	 1.3531862e-01	[ 1.3452220e+00]	 1.8269038e-01
      75	 1.4205888e+00	 1.2654478e-01	 1.4725098e+00	 1.3616568e-01	  1.3429292e+00 	 1.8693781e-01


.. parsed-literal::

      76	 1.4248813e+00	 1.2643117e-01	 1.4767710e+00	 1.3621118e-01	[ 1.3498732e+00]	 2.1158099e-01


.. parsed-literal::

      77	 1.4304387e+00	 1.2629315e-01	 1.4824842e+00	 1.3621311e-01	[ 1.3570550e+00]	 2.1626663e-01


.. parsed-literal::

      78	 1.4372536e+00	 1.2583091e-01	 1.4893875e+00	 1.3585496e-01	[ 1.3640337e+00]	 2.0455503e-01
      79	 1.4426891e+00	 1.2524630e-01	 1.4948374e+00	 1.3547945e-01	[ 1.3680773e+00]	 1.7883492e-01


.. parsed-literal::

      80	 1.4467827e+00	 1.2494552e-01	 1.4988291e+00	 1.3544952e-01	[ 1.3705481e+00]	 2.1291828e-01
      81	 1.4503036e+00	 1.2471278e-01	 1.5023356e+00	 1.3548609e-01	  1.3696208e+00 	 1.9952011e-01


.. parsed-literal::

      82	 1.4553338e+00	 1.2435365e-01	 1.5075148e+00	 1.3503520e-01	[ 1.3719857e+00]	 2.0323634e-01


.. parsed-literal::

      83	 1.4604758e+00	 1.2398945e-01	 1.5128524e+00	 1.3466620e-01	[ 1.3758344e+00]	 2.1196604e-01
      84	 1.4642043e+00	 1.2396350e-01	 1.5165113e+00	 1.3415237e-01	[ 1.3817835e+00]	 2.0684624e-01


.. parsed-literal::

      85	 1.4670104e+00	 1.2391532e-01	 1.5192395e+00	 1.3378605e-01	[ 1.3867948e+00]	 2.0576763e-01
      86	 1.4699577e+00	 1.2368439e-01	 1.5221904e+00	 1.3317267e-01	[ 1.3884716e+00]	 1.9977856e-01


.. parsed-literal::

      87	 1.4737463e+00	 1.2341705e-01	 1.5259960e+00	 1.3274967e-01	  1.3878864e+00 	 2.0436358e-01
      88	 1.4760236e+00	 1.2300915e-01	 1.5285678e+00	 1.3233971e-01	  1.3773360e+00 	 1.8435669e-01


.. parsed-literal::

      89	 1.4801619e+00	 1.2288213e-01	 1.5325903e+00	 1.3223779e-01	  1.3827447e+00 	 1.9741702e-01
      90	 1.4818171e+00	 1.2285051e-01	 1.5342443e+00	 1.3236842e-01	  1.3839102e+00 	 1.8510842e-01


.. parsed-literal::

      91	 1.4848109e+00	 1.2278142e-01	 1.5373431e+00	 1.3246800e-01	  1.3851741e+00 	 1.9851828e-01


.. parsed-literal::

      92	 1.4866381e+00	 1.2268464e-01	 1.5392395e+00	 1.3266471e-01	  1.3821902e+00 	 3.2323503e-01


.. parsed-literal::

      93	 1.4887387e+00	 1.2266057e-01	 1.5413334e+00	 1.3269174e-01	  1.3825375e+00 	 2.1231651e-01
      94	 1.4910510e+00	 1.2258173e-01	 1.5436087e+00	 1.3251674e-01	  1.3815302e+00 	 1.9927025e-01


.. parsed-literal::

      95	 1.4930551e+00	 1.2250384e-01	 1.5455582e+00	 1.3250039e-01	  1.3798018e+00 	 2.0779061e-01
      96	 1.4953898e+00	 1.2240158e-01	 1.5478484e+00	 1.3229015e-01	  1.3798829e+00 	 1.9167542e-01


.. parsed-literal::

      97	 1.4980535e+00	 1.2223752e-01	 1.5505678e+00	 1.3183201e-01	  1.3781701e+00 	 2.0837784e-01


.. parsed-literal::

      98	 1.5002334e+00	 1.2217331e-01	 1.5529641e+00	 1.3169694e-01	  1.3786108e+00 	 2.1899271e-01
      99	 1.5026624e+00	 1.2210135e-01	 1.5554203e+00	 1.3144106e-01	  1.3788848e+00 	 1.9458532e-01


.. parsed-literal::

     100	 1.5050992e+00	 1.2209239e-01	 1.5579527e+00	 1.3127310e-01	  1.3773254e+00 	 2.1271372e-01


.. parsed-literal::

     101	 1.5072878e+00	 1.2211691e-01	 1.5602137e+00	 1.3129117e-01	  1.3743993e+00 	 2.1868849e-01


.. parsed-literal::

     102	 1.5082905e+00	 1.2219955e-01	 1.5614071e+00	 1.3102604e-01	  1.3648389e+00 	 2.1382380e-01
     103	 1.5112755e+00	 1.2214974e-01	 1.5642578e+00	 1.3126921e-01	  1.3673418e+00 	 1.9815707e-01


.. parsed-literal::

     104	 1.5122442e+00	 1.2209286e-01	 1.5651855e+00	 1.3127969e-01	  1.3694412e+00 	 2.0711613e-01


.. parsed-literal::

     105	 1.5139554e+00	 1.2198439e-01	 1.5669123e+00	 1.3124627e-01	  1.3697319e+00 	 2.0986247e-01
     106	 1.5151479e+00	 1.2186923e-01	 1.5682908e+00	 1.3117302e-01	  1.3728344e+00 	 1.8613505e-01


.. parsed-literal::

     107	 1.5173185e+00	 1.2176028e-01	 1.5704740e+00	 1.3109489e-01	  1.3709935e+00 	 2.1605849e-01


.. parsed-literal::

     108	 1.5185748e+00	 1.2168999e-01	 1.5718052e+00	 1.3102982e-01	  1.3686072e+00 	 2.0284033e-01
     109	 1.5197279e+00	 1.2162195e-01	 1.5730186e+00	 1.3099455e-01	  1.3670413e+00 	 1.8906116e-01


.. parsed-literal::

     110	 1.5216992e+00	 1.2150424e-01	 1.5750623e+00	 1.3096383e-01	  1.3618853e+00 	 2.1699953e-01


.. parsed-literal::

     111	 1.5231646e+00	 1.2143183e-01	 1.5765877e+00	 1.3098442e-01	  1.3586048e+00 	 3.3212948e-01


.. parsed-literal::

     112	 1.5247922e+00	 1.2134405e-01	 1.5782025e+00	 1.3095081e-01	  1.3551198e+00 	 2.1889806e-01


.. parsed-literal::

     113	 1.5261367e+00	 1.2125149e-01	 1.5795160e+00	 1.3084566e-01	  1.3533684e+00 	 2.1212220e-01


.. parsed-literal::

     114	 1.5277346e+00	 1.2113279e-01	 1.5811085e+00	 1.3065536e-01	  1.3522196e+00 	 2.1012259e-01


.. parsed-literal::

     115	 1.5300667e+00	 1.2091630e-01	 1.5835652e+00	 1.3012522e-01	  1.3484745e+00 	 2.1751356e-01


.. parsed-literal::

     116	 1.5318305e+00	 1.2086413e-01	 1.5854504e+00	 1.2990046e-01	  1.3478242e+00 	 2.1041942e-01
     117	 1.5332776e+00	 1.2083359e-01	 1.5868689e+00	 1.2991697e-01	  1.3495046e+00 	 1.8058991e-01


.. parsed-literal::

     118	 1.5344768e+00	 1.2081053e-01	 1.5881274e+00	 1.2987104e-01	  1.3486390e+00 	 2.0504546e-01


.. parsed-literal::

     119	 1.5357262e+00	 1.2074101e-01	 1.5894821e+00	 1.2993948e-01	  1.3452289e+00 	 2.1045470e-01


.. parsed-literal::

     120	 1.5375512e+00	 1.2066217e-01	 1.5914143e+00	 1.2994171e-01	  1.3418432e+00 	 2.2505808e-01
     121	 1.5391548e+00	 1.2044275e-01	 1.5930725e+00	 1.2981920e-01	  1.3357325e+00 	 1.9899082e-01


.. parsed-literal::

     122	 1.5406871e+00	 1.2026337e-01	 1.5945635e+00	 1.2988482e-01	  1.3384998e+00 	 2.0050025e-01


.. parsed-literal::

     123	 1.5418039e+00	 1.2014973e-01	 1.5955750e+00	 1.2978163e-01	  1.3410058e+00 	 2.1048355e-01


.. parsed-literal::

     124	 1.5435439e+00	 1.1991941e-01	 1.5972234e+00	 1.2969136e-01	  1.3435939e+00 	 2.1902394e-01


.. parsed-literal::

     125	 1.5443292e+00	 1.1976801e-01	 1.5980272e+00	 1.2963711e-01	  1.3457114e+00 	 2.1008253e-01


.. parsed-literal::

     126	 1.5452354e+00	 1.1978192e-01	 1.5989452e+00	 1.2962486e-01	  1.3461896e+00 	 2.0891476e-01
     127	 1.5462329e+00	 1.1977791e-01	 1.6000182e+00	 1.2960525e-01	  1.3451449e+00 	 1.9937396e-01


.. parsed-literal::

     128	 1.5472655e+00	 1.1975624e-01	 1.6011108e+00	 1.2950998e-01	  1.3437934e+00 	 2.0461464e-01


.. parsed-literal::

     129	 1.5488662e+00	 1.1971261e-01	 1.6028350e+00	 1.2923245e-01	  1.3435337e+00 	 2.0541787e-01


.. parsed-literal::

     130	 1.5500006e+00	 1.1959345e-01	 1.6040304e+00	 1.2884596e-01	  1.3377803e+00 	 2.1478438e-01


.. parsed-literal::

     131	 1.5510042e+00	 1.1959177e-01	 1.6049306e+00	 1.2887430e-01	  1.3418172e+00 	 2.0982885e-01


.. parsed-literal::

     132	 1.5518586e+00	 1.1958254e-01	 1.6056977e+00	 1.2893752e-01	  1.3441409e+00 	 2.1274877e-01
     133	 1.5528926e+00	 1.1954120e-01	 1.6066924e+00	 1.2887429e-01	  1.3435733e+00 	 1.8728900e-01


.. parsed-literal::

     134	 1.5542015e+00	 1.1952663e-01	 1.6079809e+00	 1.2895065e-01	  1.3410714e+00 	 2.1975040e-01
     135	 1.5554855e+00	 1.1952260e-01	 1.6093147e+00	 1.2902090e-01	  1.3363245e+00 	 2.0051479e-01


.. parsed-literal::

     136	 1.5566365e+00	 1.1951167e-01	 1.6105541e+00	 1.2886015e-01	  1.3302186e+00 	 2.1507215e-01
     137	 1.5579415e+00	 1.1953282e-01	 1.6119186e+00	 1.2890452e-01	  1.3264822e+00 	 1.7651749e-01


.. parsed-literal::

     138	 1.5592137e+00	 1.1953190e-01	 1.6132352e+00	 1.2887573e-01	  1.3242083e+00 	 1.7841673e-01
     139	 1.5605765e+00	 1.1950757e-01	 1.6147320e+00	 1.2893791e-01	  1.3168711e+00 	 2.0277119e-01


.. parsed-literal::

     140	 1.5617949e+00	 1.1951783e-01	 1.6159784e+00	 1.2914203e-01	  1.3140410e+00 	 1.7736053e-01


.. parsed-literal::

     141	 1.5627129e+00	 1.1949628e-01	 1.6168630e+00	 1.2923354e-01	  1.3132674e+00 	 2.1864462e-01
     142	 1.5643751e+00	 1.1952668e-01	 1.6185426e+00	 1.2945790e-01	  1.3087253e+00 	 1.8277383e-01


.. parsed-literal::

     143	 1.5650646e+00	 1.1951841e-01	 1.6192723e+00	 1.2954463e-01	  1.3056762e+00 	 2.1002913e-01
     144	 1.5661727e+00	 1.1953003e-01	 1.6203389e+00	 1.2951521e-01	  1.3083092e+00 	 1.7822576e-01


.. parsed-literal::

     145	 1.5671540e+00	 1.1955037e-01	 1.6213287e+00	 1.2950053e-01	  1.3097393e+00 	 1.8833542e-01


.. parsed-literal::

     146	 1.5680744e+00	 1.1954479e-01	 1.6222580e+00	 1.2951662e-01	  1.3109771e+00 	 2.1201944e-01
     147	 1.5688086e+00	 1.1953200e-01	 1.6230795e+00	 1.2978118e-01	  1.3081935e+00 	 1.7569852e-01


.. parsed-literal::

     148	 1.5701446e+00	 1.1946931e-01	 1.6243378e+00	 1.2979663e-01	  1.3111041e+00 	 1.8998885e-01


.. parsed-literal::

     149	 1.5709314e+00	 1.1940228e-01	 1.6250891e+00	 1.2984114e-01	  1.3116936e+00 	 2.1753097e-01


.. parsed-literal::

     150	 1.5720161e+00	 1.1932397e-01	 1.6261510e+00	 1.2991824e-01	  1.3116237e+00 	 2.2061419e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.22 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f11b70fd360>



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
    CPU times: user 1.76 s, sys: 55 ms, total: 1.82 s
    Wall time: 600 ms


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

