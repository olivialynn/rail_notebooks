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
       1	-3.2628909e-01	 3.1537225e-01	-3.1647380e-01	 3.4360261e-01	[-3.6924938e-01]	 4.6419644e-01


.. parsed-literal::

       2	-2.5807683e-01	 3.0567284e-01	-2.3507102e-01	 3.3074657e-01	[-3.0930236e-01]	 2.2931409e-01


.. parsed-literal::

       3	-2.1246498e-01	 2.8465026e-01	-1.7071476e-01	 3.0743923e-01	[-2.5771099e-01]	 2.7427959e-01


.. parsed-literal::

       4	-1.9278515e-01	 2.6358379e-01	-1.5237350e-01	 2.7875225e-01	[-2.3937806e-01]	 2.0743704e-01
       5	-1.0276248e-01	 2.5685715e-01	-6.8250706e-02	 2.7167501e-01	[-1.2808248e-01]	 1.9825101e-01


.. parsed-literal::

       6	-6.7808283e-02	 2.5021724e-01	-3.5129368e-02	 2.6122693e-01	[-7.5786784e-02]	 2.1270132e-01


.. parsed-literal::

       7	-4.7461770e-02	 2.4732149e-01	-2.2436681e-02	 2.5806608e-01	[-6.4299696e-02]	 2.0473623e-01
       8	-2.9946369e-02	 2.4417494e-01	-9.1689933e-03	 2.5612247e-01	[-5.7182232e-02]	 1.7433548e-01


.. parsed-literal::

       9	-1.2796043e-02	 2.4104108e-01	 4.4972670e-03	 2.5466664e-01	[-5.3881555e-02]	 2.1029425e-01
      10	-6.0933424e-03	 2.4019519e-01	 8.8286130e-03	 2.5489523e-01	[-5.0337194e-02]	 1.9718313e-01


.. parsed-literal::

      11	 1.9926709e-04	 2.3879687e-01	 1.4766163e-02	 2.5225090e-01	[-4.3201204e-02]	 1.7486548e-01


.. parsed-literal::

      12	 3.1422167e-03	 2.3823418e-01	 1.7450209e-02	 2.5171380e-01	[-4.0778060e-02]	 2.1073866e-01
      13	 6.7000555e-03	 2.3755022e-01	 2.0956511e-02	 2.5098876e-01	[-3.6875052e-02]	 1.9654918e-01


.. parsed-literal::

      14	 7.0986839e-02	 2.2832350e-01	 8.8576522e-02	 2.4150263e-01	[ 3.6983909e-02]	 3.2707500e-01


.. parsed-literal::

      15	 7.3361014e-02	 2.2354681e-01	 9.4844633e-02	 2.3939414e-01	[ 7.0303145e-02]	 3.4332395e-01


.. parsed-literal::

      16	 2.2982424e-01	 2.2151685e-01	 2.6156123e-01	 2.3978189e-01	[ 2.1430384e-01]	 2.1144700e-01


.. parsed-literal::

      17	 2.9301063e-01	 2.1154054e-01	 3.2553437e-01	 2.2337814e-01	[ 2.9056602e-01]	 2.2029996e-01


.. parsed-literal::

      18	 3.4259560e-01	 2.0432241e-01	 3.7530290e-01	 2.1581217e-01	[ 3.4717242e-01]	 2.0984316e-01


.. parsed-literal::

      19	 3.9999585e-01	 1.9981193e-01	 4.3374148e-01	 2.1361357e-01	[ 4.0563613e-01]	 2.0736361e-01


.. parsed-literal::

      20	 5.2996874e-01	 1.9500872e-01	 5.6526699e-01	 2.0432094e-01	[ 5.5284573e-01]	 2.0675087e-01


.. parsed-literal::

      21	 6.0925580e-01	 1.9522752e-01	 6.4792036e-01	 2.0451905e-01	[ 6.5152511e-01]	 2.0593762e-01


.. parsed-literal::

      22	 6.4254357e-01	 1.8970635e-01	 6.8083705e-01	 1.9787467e-01	[ 6.7850910e-01]	 2.1221519e-01


.. parsed-literal::

      23	 6.7188949e-01	 1.8908563e-01	 7.1054093e-01	 1.9881975e-01	[ 6.9591238e-01]	 2.0181179e-01


.. parsed-literal::

      24	 7.0305803e-01	 1.9066665e-01	 7.4134483e-01	 2.0126892e-01	[ 7.2372698e-01]	 2.0561695e-01


.. parsed-literal::

      25	 7.4445450e-01	 1.8889392e-01	 7.8237550e-01	 2.0465354e-01	[ 7.5033859e-01]	 2.1607256e-01


.. parsed-literal::

      26	 7.8864014e-01	 1.9068976e-01	 8.2724523e-01	 2.1179818e-01	[ 7.8817626e-01]	 2.0535541e-01


.. parsed-literal::

      27	 8.3035623e-01	 1.8699369e-01	 8.7115401e-01	 2.0677472e-01	[ 8.3363554e-01]	 2.1210098e-01


.. parsed-literal::

      28	 8.6398636e-01	 1.8319437e-01	 9.0579359e-01	 2.0143165e-01	[ 8.5694058e-01]	 2.1150398e-01


.. parsed-literal::

      29	 8.9283359e-01	 1.8054753e-01	 9.3640330e-01	 1.9906059e-01	[ 8.7403637e-01]	 2.0360184e-01
      30	 9.1505724e-01	 1.7729774e-01	 9.5853414e-01	 1.9290688e-01	[ 8.9817635e-01]	 1.6559434e-01


.. parsed-literal::

      31	 9.4588621e-01	 1.7132280e-01	 9.9073534e-01	 1.8997020e-01	[ 9.1622116e-01]	 2.1231890e-01


.. parsed-literal::

      32	 9.7089071e-01	 1.6616302e-01	 1.0159344e+00	 1.8424457e-01	[ 9.3108060e-01]	 2.1191835e-01


.. parsed-literal::

      33	 9.9150022e-01	 1.6272793e-01	 1.0366222e+00	 1.8091805e-01	[ 9.4824383e-01]	 2.1730852e-01


.. parsed-literal::

      34	 1.0097074e+00	 1.5957608e-01	 1.0549555e+00	 1.7645033e-01	[ 9.6275620e-01]	 2.0768285e-01
      35	 1.0270205e+00	 1.5819031e-01	 1.0727049e+00	 1.7546051e-01	[ 9.8558683e-01]	 1.9679761e-01


.. parsed-literal::

      36	 1.0456238e+00	 1.5662760e-01	 1.0926624e+00	 1.7453903e-01	[ 1.0019349e+00]	 2.1542835e-01


.. parsed-literal::

      37	 1.0608842e+00	 1.5629315e-01	 1.1085902e+00	 1.7575565e-01	[ 1.0172157e+00]	 2.0675468e-01


.. parsed-literal::

      38	 1.0727928e+00	 1.5558654e-01	 1.1203778e+00	 1.7703989e-01	[ 1.0300406e+00]	 2.0808983e-01


.. parsed-literal::

      39	 1.0869292e+00	 1.5431758e-01	 1.1347035e+00	 1.7628449e-01	[ 1.0386256e+00]	 2.0804739e-01
      40	 1.0987562e+00	 1.5250002e-01	 1.1466076e+00	 1.7468968e-01	[ 1.0505723e+00]	 1.7544174e-01


.. parsed-literal::

      41	 1.1107633e+00	 1.5025247e-01	 1.1585348e+00	 1.7131634e-01	[ 1.0577287e+00]	 2.1658230e-01


.. parsed-literal::

      42	 1.1256059e+00	 1.4719675e-01	 1.1733827e+00	 1.6703700e-01	[ 1.0745805e+00]	 2.0792484e-01
      43	 1.1397162e+00	 1.4554942e-01	 1.1879174e+00	 1.6569123e-01	[ 1.0827764e+00]	 1.8847251e-01


.. parsed-literal::

      44	 1.1525405e+00	 1.4442481e-01	 1.2006517e+00	 1.6431121e-01	[ 1.0981694e+00]	 2.0680261e-01
      45	 1.1670922e+00	 1.4334627e-01	 1.2154552e+00	 1.6378300e-01	[ 1.1161354e+00]	 2.0388389e-01


.. parsed-literal::

      46	 1.1776021e+00	 1.4260590e-01	 1.2261817e+00	 1.6227989e-01	[ 1.1279962e+00]	 2.0676994e-01
      47	 1.1879451e+00	 1.4212321e-01	 1.2366477e+00	 1.6121673e-01	[ 1.1407708e+00]	 1.9727516e-01


.. parsed-literal::

      48	 1.1979402e+00	 1.4157801e-01	 1.2468156e+00	 1.6063021e-01	[ 1.1441053e+00]	 1.8373227e-01
      49	 1.2091140e+00	 1.4085450e-01	 1.2582924e+00	 1.6008808e-01	[ 1.1497724e+00]	 2.0405221e-01


.. parsed-literal::

      50	 1.2185956e+00	 1.3990694e-01	 1.2680052e+00	 1.5916608e-01	[ 1.1537810e+00]	 2.0647049e-01
      51	 1.2267691e+00	 1.3940949e-01	 1.2761719e+00	 1.5862943e-01	[ 1.1633177e+00]	 1.9146109e-01


.. parsed-literal::

      52	 1.2397218e+00	 1.3869319e-01	 1.2896948e+00	 1.5658991e-01	[ 1.1809421e+00]	 2.1061826e-01
      53	 1.2492456e+00	 1.3887301e-01	 1.2995766e+00	 1.5711177e-01	[ 1.1954611e+00]	 1.8510389e-01


.. parsed-literal::

      54	 1.2582819e+00	 1.3842804e-01	 1.3085842e+00	 1.5602247e-01	[ 1.2079405e+00]	 1.9796705e-01


.. parsed-literal::

      55	 1.2656177e+00	 1.3812279e-01	 1.3158733e+00	 1.5546097e-01	[ 1.2145000e+00]	 2.0666599e-01
      56	 1.2756801e+00	 1.3760664e-01	 1.3261185e+00	 1.5459747e-01	[ 1.2202330e+00]	 1.7303276e-01


.. parsed-literal::

      57	 1.2845412e+00	 1.3719025e-01	 1.3348479e+00	 1.5323137e-01	[ 1.2281790e+00]	 2.0387697e-01
      58	 1.2933995e+00	 1.3636907e-01	 1.3438409e+00	 1.5182385e-01	[ 1.2363823e+00]	 2.0126319e-01


.. parsed-literal::

      59	 1.2993160e+00	 1.3610723e-01	 1.3497952e+00	 1.5126990e-01	[ 1.2413144e+00]	 1.9949651e-01


.. parsed-literal::

      60	 1.3071529e+00	 1.3562358e-01	 1.3579029e+00	 1.4984628e-01	[ 1.2461763e+00]	 2.0879483e-01


.. parsed-literal::

      61	 1.3158521e+00	 1.3533882e-01	 1.3667654e+00	 1.4829306e-01	[ 1.2472805e+00]	 2.0809317e-01


.. parsed-literal::

      62	 1.3240927e+00	 1.3486408e-01	 1.3749755e+00	 1.4799095e-01	[ 1.2521716e+00]	 2.0763516e-01


.. parsed-literal::

      63	 1.3306717e+00	 1.3448396e-01	 1.3816673e+00	 1.4801005e-01	[ 1.2549488e+00]	 2.0556116e-01


.. parsed-literal::

      64	 1.3367826e+00	 1.3417273e-01	 1.3879627e+00	 1.4877241e-01	[ 1.2591646e+00]	 2.0992875e-01
      65	 1.3445047e+00	 1.3380995e-01	 1.3960305e+00	 1.4920098e-01	[ 1.2635406e+00]	 1.9849658e-01


.. parsed-literal::

      66	 1.3510360e+00	 1.3356848e-01	 1.4026725e+00	 1.4929773e-01	[ 1.2676342e+00]	 1.9992542e-01


.. parsed-literal::

      67	 1.3559800e+00	 1.3306317e-01	 1.4076479e+00	 1.4850474e-01	[ 1.2737377e+00]	 2.0904970e-01


.. parsed-literal::

      68	 1.3625877e+00	 1.3226969e-01	 1.4145029e+00	 1.4732072e-01	[ 1.2775562e+00]	 2.1080542e-01
      69	 1.3687918e+00	 1.3148051e-01	 1.4210696e+00	 1.4570824e-01	[ 1.2839983e+00]	 1.6945863e-01


.. parsed-literal::

      70	 1.3742672e+00	 1.3109689e-01	 1.4264105e+00	 1.4532328e-01	[ 1.2904160e+00]	 1.8311453e-01
      71	 1.3788805e+00	 1.3098566e-01	 1.4309874e+00	 1.4537640e-01	[ 1.2933445e+00]	 1.9733000e-01


.. parsed-literal::

      72	 1.3850390e+00	 1.3085278e-01	 1.4372833e+00	 1.4542205e-01	[ 1.2962956e+00]	 1.9853544e-01
      73	 1.3864962e+00	 1.3161578e-01	 1.4389690e+00	 1.4520616e-01	  1.2947341e+00 	 1.6628003e-01


.. parsed-literal::

      74	 1.3930528e+00	 1.3102042e-01	 1.4453289e+00	 1.4518058e-01	[ 1.2991606e+00]	 1.7198515e-01


.. parsed-literal::

      75	 1.3954187e+00	 1.3088737e-01	 1.4477541e+00	 1.4496565e-01	[ 1.2996406e+00]	 2.0657635e-01
      76	 1.4000639e+00	 1.3081082e-01	 1.4525087e+00	 1.4487904e-01	[ 1.3002904e+00]	 1.9872904e-01


.. parsed-literal::

      77	 1.4039587e+00	 1.3112397e-01	 1.4566698e+00	 1.4448753e-01	  1.2979108e+00 	 2.0032477e-01


.. parsed-literal::

      78	 1.4092685e+00	 1.3116282e-01	 1.4619469e+00	 1.4462768e-01	[ 1.3006098e+00]	 2.1125722e-01
      79	 1.4132069e+00	 1.3113737e-01	 1.4658828e+00	 1.4489533e-01	[ 1.3026624e+00]	 1.8937135e-01


.. parsed-literal::

      80	 1.4176988e+00	 1.3108726e-01	 1.4704863e+00	 1.4518221e-01	[ 1.3037230e+00]	 2.0825791e-01


.. parsed-literal::

      81	 1.4221686e+00	 1.3079075e-01	 1.4750748e+00	 1.4494474e-01	  1.3021922e+00 	 2.0829606e-01


.. parsed-literal::

      82	 1.4268112e+00	 1.3058481e-01	 1.4797273e+00	 1.4497278e-01	[ 1.3049255e+00]	 2.1457934e-01
      83	 1.4296471e+00	 1.3035599e-01	 1.4825714e+00	 1.4472830e-01	  1.3044063e+00 	 1.9695807e-01


.. parsed-literal::

      84	 1.4329153e+00	 1.3008084e-01	 1.4859657e+00	 1.4429518e-01	  1.2996989e+00 	 2.1341109e-01


.. parsed-literal::

      85	 1.4354021e+00	 1.2997103e-01	 1.4886742e+00	 1.4470892e-01	  1.2985378e+00 	 2.1852326e-01


.. parsed-literal::

      86	 1.4381287e+00	 1.2991495e-01	 1.4913174e+00	 1.4494096e-01	  1.3005249e+00 	 2.1928096e-01


.. parsed-literal::

      87	 1.4417031e+00	 1.3009046e-01	 1.4948731e+00	 1.4582112e-01	[ 1.3053028e+00]	 2.1407318e-01


.. parsed-literal::

      88	 1.4446181e+00	 1.3013965e-01	 1.4977417e+00	 1.4653627e-01	[ 1.3081200e+00]	 2.0533156e-01


.. parsed-literal::

      89	 1.4484210e+00	 1.3074945e-01	 1.5015827e+00	 1.4843633e-01	[ 1.3121337e+00]	 2.1059775e-01


.. parsed-literal::

      90	 1.4515588e+00	 1.3031505e-01	 1.5047006e+00	 1.4811050e-01	  1.3094873e+00 	 2.0686674e-01
      91	 1.4532462e+00	 1.3011537e-01	 1.5064195e+00	 1.4761779e-01	  1.3068633e+00 	 1.9933128e-01


.. parsed-literal::

      92	 1.4576984e+00	 1.2978208e-01	 1.5111001e+00	 1.4700623e-01	  1.3029809e+00 	 2.0460939e-01


.. parsed-literal::

      93	 1.4596467e+00	 1.2964614e-01	 1.5132704e+00	 1.4742817e-01	  1.2939772e+00 	 2.1068740e-01


.. parsed-literal::

      94	 1.4633489e+00	 1.2968752e-01	 1.5168297e+00	 1.4766355e-01	  1.3068856e+00 	 2.1015501e-01


.. parsed-literal::

      95	 1.4654126e+00	 1.2978803e-01	 1.5188390e+00	 1.4826633e-01	[ 1.3149998e+00]	 2.1781397e-01


.. parsed-literal::

      96	 1.4673748e+00	 1.2981492e-01	 1.5208011e+00	 1.4860041e-01	[ 1.3176675e+00]	 2.1927547e-01


.. parsed-literal::

      97	 1.4701928e+00	 1.3005467e-01	 1.5237341e+00	 1.4955851e-01	  1.3168342e+00 	 2.1512580e-01


.. parsed-literal::

      98	 1.4732009e+00	 1.2995965e-01	 1.5267636e+00	 1.4934620e-01	  1.3100118e+00 	 2.1080780e-01


.. parsed-literal::

      99	 1.4753757e+00	 1.2985069e-01	 1.5289809e+00	 1.4909290e-01	  1.3055222e+00 	 2.1338177e-01
     100	 1.4779019e+00	 1.2966920e-01	 1.5316228e+00	 1.4880129e-01	  1.3003742e+00 	 1.8499351e-01


.. parsed-literal::

     101	 1.4799488e+00	 1.2956336e-01	 1.5337959e+00	 1.4875882e-01	  1.2989154e+00 	 1.8336391e-01
     102	 1.4824481e+00	 1.2943536e-01	 1.5362661e+00	 1.4880519e-01	  1.2993200e+00 	 2.0324969e-01


.. parsed-literal::

     103	 1.4846109e+00	 1.2937783e-01	 1.5383610e+00	 1.4896003e-01	  1.3017858e+00 	 1.8480539e-01


.. parsed-literal::

     104	 1.4864035e+00	 1.2938151e-01	 1.5401207e+00	 1.4921231e-01	  1.3016004e+00 	 2.0772719e-01


.. parsed-literal::

     105	 1.4892024e+00	 1.2900549e-01	 1.5430108e+00	 1.4939918e-01	  1.2895289e+00 	 2.0841813e-01
     106	 1.4921857e+00	 1.2901381e-01	 1.5460123e+00	 1.4974117e-01	  1.2908245e+00 	 2.0715070e-01


.. parsed-literal::

     107	 1.4938495e+00	 1.2886185e-01	 1.5476924e+00	 1.4956786e-01	  1.2899436e+00 	 1.9055247e-01


.. parsed-literal::

     108	 1.4962592e+00	 1.2847448e-01	 1.5502536e+00	 1.4914298e-01	  1.2861137e+00 	 2.1050858e-01


.. parsed-literal::

     109	 1.4966888e+00	 1.2832657e-01	 1.5509705e+00	 1.4928008e-01	  1.2828532e+00 	 2.1129322e-01


.. parsed-literal::

     110	 1.4994106e+00	 1.2817176e-01	 1.5535425e+00	 1.4894804e-01	  1.2851462e+00 	 2.1863008e-01
     111	 1.5005903e+00	 1.2808641e-01	 1.5547127e+00	 1.4885372e-01	  1.2853995e+00 	 1.9810462e-01


.. parsed-literal::

     112	 1.5019285e+00	 1.2800067e-01	 1.5560613e+00	 1.4866113e-01	  1.2853257e+00 	 2.0322847e-01
     113	 1.5037832e+00	 1.2784412e-01	 1.5579368e+00	 1.4832158e-01	  1.2822229e+00 	 1.9876003e-01


.. parsed-literal::

     114	 1.5050672e+00	 1.2774039e-01	 1.5594271e+00	 1.4750863e-01	  1.2758844e+00 	 2.0664048e-01
     115	 1.5070010e+00	 1.2761874e-01	 1.5613013e+00	 1.4732999e-01	  1.2741448e+00 	 1.9050026e-01


.. parsed-literal::

     116	 1.5078973e+00	 1.2758165e-01	 1.5622214e+00	 1.4719886e-01	  1.2724928e+00 	 2.0146823e-01


.. parsed-literal::

     117	 1.5098074e+00	 1.2750099e-01	 1.5642348e+00	 1.4678869e-01	  1.2681471e+00 	 2.1393991e-01
     118	 1.5111820e+00	 1.2752460e-01	 1.5659033e+00	 1.4598695e-01	  1.2608392e+00 	 1.7981815e-01


.. parsed-literal::

     119	 1.5136455e+00	 1.2741556e-01	 1.5682977e+00	 1.4576088e-01	  1.2609846e+00 	 2.1226215e-01
     120	 1.5149381e+00	 1.2734265e-01	 1.5695329e+00	 1.4565353e-01	  1.2633535e+00 	 1.8615055e-01


.. parsed-literal::

     121	 1.5164531e+00	 1.2727224e-01	 1.5709959e+00	 1.4552362e-01	  1.2643346e+00 	 1.7457557e-01


.. parsed-literal::

     122	 1.5186784e+00	 1.2708576e-01	 1.5731593e+00	 1.4543429e-01	  1.2620541e+00 	 2.1315265e-01


.. parsed-literal::

     123	 1.5207207e+00	 1.2706390e-01	 1.5751466e+00	 1.4561977e-01	  1.2613449e+00 	 2.1085477e-01


.. parsed-literal::

     124	 1.5220488e+00	 1.2706444e-01	 1.5765012e+00	 1.4572848e-01	  1.2567782e+00 	 2.1098614e-01


.. parsed-literal::

     125	 1.5236039e+00	 1.2700647e-01	 1.5781526e+00	 1.4581407e-01	  1.2535489e+00 	 2.0758295e-01


.. parsed-literal::

     126	 1.5245800e+00	 1.2689784e-01	 1.5793189e+00	 1.4592316e-01	  1.2475340e+00 	 2.1250558e-01


.. parsed-literal::

     127	 1.5261143e+00	 1.2677478e-01	 1.5808330e+00	 1.4571514e-01	  1.2503058e+00 	 2.0754933e-01


.. parsed-literal::

     128	 1.5269526e+00	 1.2663897e-01	 1.5816733e+00	 1.4558906e-01	  1.2525470e+00 	 2.0969677e-01


.. parsed-literal::

     129	 1.5276417e+00	 1.2653481e-01	 1.5823745e+00	 1.4554211e-01	  1.2528296e+00 	 2.0896864e-01
     130	 1.5289385e+00	 1.2639791e-01	 1.5836901e+00	 1.4560229e-01	  1.2523928e+00 	 1.8054533e-01


.. parsed-literal::

     131	 1.5300328e+00	 1.2619625e-01	 1.5848012e+00	 1.4559743e-01	  1.2496087e+00 	 3.1179214e-01
     132	 1.5316569e+00	 1.2614200e-01	 1.5864187e+00	 1.4572608e-01	  1.2481914e+00 	 1.9774365e-01


.. parsed-literal::

     133	 1.5332402e+00	 1.2607593e-01	 1.5880282e+00	 1.4578638e-01	  1.2450730e+00 	 2.0867205e-01
     134	 1.5343533e+00	 1.2601698e-01	 1.5891972e+00	 1.4573586e-01	  1.2424760e+00 	 1.8416476e-01


.. parsed-literal::

     135	 1.5352915e+00	 1.2575892e-01	 1.5903195e+00	 1.4544559e-01	  1.2330631e+00 	 2.1048689e-01


.. parsed-literal::

     136	 1.5365859e+00	 1.2573611e-01	 1.5915992e+00	 1.4539499e-01	  1.2345364e+00 	 2.0597053e-01


.. parsed-literal::

     137	 1.5374308e+00	 1.2564291e-01	 1.5924842e+00	 1.4531693e-01	  1.2335508e+00 	 2.0968151e-01
     138	 1.5382520e+00	 1.2556661e-01	 1.5933562e+00	 1.4532976e-01	  1.2309051e+00 	 1.7264199e-01


.. parsed-literal::

     139	 1.5386504e+00	 1.2536442e-01	 1.5938824e+00	 1.4531219e-01	  1.2235504e+00 	 2.1134949e-01


.. parsed-literal::

     140	 1.5400130e+00	 1.2545390e-01	 1.5951822e+00	 1.4553566e-01	  1.2212603e+00 	 2.0815420e-01
     141	 1.5406237e+00	 1.2549574e-01	 1.5957677e+00	 1.4565720e-01	  1.2198134e+00 	 2.0621181e-01


.. parsed-literal::

     142	 1.5414824e+00	 1.2549406e-01	 1.5966112e+00	 1.4580203e-01	  1.2165827e+00 	 2.1608281e-01


.. parsed-literal::

     143	 1.5428364e+00	 1.2535557e-01	 1.5979312e+00	 1.4570734e-01	  1.2204306e+00 	 2.1223330e-01


.. parsed-literal::

     144	 1.5442624e+00	 1.2507361e-01	 1.5993663e+00	 1.4565032e-01	  1.2148803e+00 	 2.1597958e-01
     145	 1.5453631e+00	 1.2491776e-01	 1.6004304e+00	 1.4544644e-01	  1.2211432e+00 	 2.0418334e-01


.. parsed-literal::

     146	 1.5462834e+00	 1.2480329e-01	 1.6013724e+00	 1.4529259e-01	  1.2227776e+00 	 2.1671343e-01


.. parsed-literal::

     147	 1.5476083e+00	 1.2464916e-01	 1.6027489e+00	 1.4501980e-01	  1.2189595e+00 	 2.1839356e-01


.. parsed-literal::

     148	 1.5486235e+00	 1.2456127e-01	 1.6038384e+00	 1.4491859e-01	  1.2121528e+00 	 3.0012488e-01
     149	 1.5498277e+00	 1.2448556e-01	 1.6050756e+00	 1.4479826e-01	  1.2055750e+00 	 1.8819427e-01


.. parsed-literal::

     150	 1.5507407e+00	 1.2447475e-01	 1.6060031e+00	 1.4481142e-01	  1.2015843e+00 	 1.9048142e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.03 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2b7c0e3880>



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
    CPU times: user 2.09 s, sys: 37.9 ms, total: 2.13 s
    Wall time: 637 ms


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

