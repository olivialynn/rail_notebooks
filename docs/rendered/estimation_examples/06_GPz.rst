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
       1	-3.4112426e-01	 3.1955159e-01	-3.3143329e-01	 3.2440041e-01	[-3.4118623e-01]	 4.6668744e-01


.. parsed-literal::

       2	-2.6754516e-01	 3.0804000e-01	-2.4279885e-01	 3.1327082e-01	[-2.5930275e-01]	 2.3193002e-01


.. parsed-literal::

       3	-2.2327187e-01	 2.8802925e-01	-1.8117777e-01	 2.9324984e-01	[-2.0600749e-01]	 2.8288317e-01


.. parsed-literal::

       4	-1.9364694e-01	 2.6359679e-01	-1.5326463e-01	 2.7107737e-01	[-2.0310861e-01]	 2.0776343e-01
       5	-1.0114241e-01	 2.5580273e-01	-6.4875154e-02	 2.6542682e-01	[-1.1302223e-01]	 1.9865084e-01


.. parsed-literal::

       6	-6.4679944e-02	 2.4956338e-01	-3.2025267e-02	 2.5870036e-01	[-6.7490154e-02]	 1.9811845e-01


.. parsed-literal::

       7	-4.6979907e-02	 2.4709386e-01	-2.1411241e-02	 2.5703754e-01	[-6.1688655e-02]	 2.0725870e-01
       8	-3.3445547e-02	 2.4490347e-01	-1.2216622e-02	 2.5511311e-01	[-5.4149024e-02]	 1.9367480e-01


.. parsed-literal::

       9	-1.8846413e-02	 2.4225961e-01	-9.6208615e-04	 2.5309201e-01	[-4.5247674e-02]	 2.0839405e-01
      10	-9.7118280e-03	 2.4076922e-01	 5.7182858e-03	 2.5102678e-01	[-3.6701017e-02]	 1.9802880e-01


.. parsed-literal::

      11	-3.9864770e-03	 2.3973699e-01	 1.0475024e-02	 2.4969863e-01	[-2.9512016e-02]	 2.0990300e-01
      12	-1.7474234e-03	 2.3929432e-01	 1.2465044e-02	 2.4910845e-01	[-2.6899531e-02]	 1.8849683e-01


.. parsed-literal::

      13	 3.2453688e-03	 2.3837710e-01	 1.7242151e-02	 2.4746264e-01	[-2.0054726e-02]	 2.1198988e-01


.. parsed-literal::

      14	 3.9992394e-02	 2.2835340e-01	 5.5901801e-02	 2.3705782e-01	[ 2.6370120e-02]	 3.3404183e-01


.. parsed-literal::

      15	 5.1053214e-02	 2.2207100e-01	 6.9369576e-02	 2.3085087e-01	[ 4.7690230e-02]	 2.1105742e-01
      16	 1.5695971e-01	 2.1656174e-01	 1.7705419e-01	 2.2471827e-01	[ 1.5211007e-01]	 2.0272660e-01


.. parsed-literal::

      17	 2.3058690e-01	 2.1081737e-01	 2.6073770e-01	 2.1995250e-01	[ 2.1133967e-01]	 1.9885302e-01
      18	 2.9495822e-01	 2.1068655e-01	 3.2581763e-01	 2.1684981e-01	[ 2.9286125e-01]	 1.8800402e-01


.. parsed-literal::

      19	 3.2535645e-01	 2.0950846e-01	 3.5605703e-01	 2.1778160e-01	[ 3.2130394e-01]	 2.0130324e-01


.. parsed-literal::

      20	 3.6267154e-01	 2.0639188e-01	 3.9285371e-01	 2.1626938e-01	[ 3.5388723e-01]	 2.1032643e-01


.. parsed-literal::

      21	 4.2286865e-01	 2.0410794e-01	 4.5442930e-01	 2.1427858e-01	[ 4.1323692e-01]	 2.0085669e-01
      22	 5.2273032e-01	 2.0455986e-01	 5.5725765e-01	 2.1417078e-01	[ 5.3293584e-01]	 1.8002701e-01


.. parsed-literal::

      23	 5.6023191e-01	 2.0509390e-01	 5.9884670e-01	 2.0874037e-01	[ 6.0174019e-01]	 2.1077251e-01


.. parsed-literal::

      24	 6.1271100e-01	 1.9893723e-01	 6.5104276e-01	 2.0408355e-01	[ 6.4857735e-01]	 2.0387149e-01


.. parsed-literal::

      25	 6.3884265e-01	 1.9616860e-01	 6.7632871e-01	 2.0120356e-01	[ 6.6839416e-01]	 2.0996547e-01


.. parsed-literal::

      26	 6.9450227e-01	 1.8931537e-01	 7.2988407e-01	 1.9404055e-01	[ 7.1866796e-01]	 2.2255969e-01


.. parsed-literal::

      27	 7.2253185e-01	 1.8574550e-01	 7.5765830e-01	 1.9321271e-01	[ 7.3925028e-01]	 3.2736015e-01


.. parsed-literal::

      28	 7.4975670e-01	 1.8587657e-01	 7.8674336e-01	 1.9796693e-01	[ 7.6771149e-01]	 2.0862675e-01


.. parsed-literal::

      29	 7.7019838e-01	 1.8420774e-01	 8.0720304e-01	 1.9761332e-01	[ 7.8680769e-01]	 2.1137643e-01


.. parsed-literal::

      30	 7.8080920e-01	 1.8443847e-01	 8.1764889e-01	 1.9949449e-01	[ 7.9175668e-01]	 2.1385813e-01


.. parsed-literal::

      31	 7.9885593e-01	 1.8864703e-01	 8.3645881e-01	 2.0401621e-01	[ 8.2179572e-01]	 2.1002126e-01


.. parsed-literal::

      32	 8.2866298e-01	 1.8785025e-01	 8.6682592e-01	 2.0392117e-01	[ 8.5073392e-01]	 2.2068095e-01


.. parsed-literal::

      33	 8.5141587e-01	 1.8624018e-01	 8.9036615e-01	 2.0266171e-01	[ 8.6796967e-01]	 2.1416640e-01


.. parsed-literal::

      34	 8.8004317e-01	 1.8172476e-01	 9.1988701e-01	 1.9689622e-01	[ 9.0139203e-01]	 2.1280551e-01


.. parsed-literal::

      35	 9.0651107e-01	 1.8003101e-01	 9.4700107e-01	 1.9438335e-01	[ 9.3874397e-01]	 2.0606542e-01


.. parsed-literal::

      36	 9.2214847e-01	 1.7949634e-01	 9.6403136e-01	 1.9491715e-01	[ 9.6036869e-01]	 2.0928860e-01


.. parsed-literal::

      37	 9.4389890e-01	 1.7812897e-01	 9.8540334e-01	 1.9418825e-01	[ 9.8011130e-01]	 2.1708035e-01
      38	 9.5826557e-01	 1.7775208e-01	 1.0001773e+00	 1.9416174e-01	[ 9.8963541e-01]	 2.0177722e-01


.. parsed-literal::

      39	 9.7809615e-01	 1.7693094e-01	 1.0217459e+00	 1.9464407e-01	[ 9.9827206e-01]	 2.0911026e-01


.. parsed-literal::

      40	 9.9112424e-01	 1.7566380e-01	 1.0356234e+00	 1.9315133e-01	[ 1.0051662e+00]	 2.1016431e-01
      41	 1.0058487e+00	 1.7361151e-01	 1.0503844e+00	 1.9084545e-01	[ 1.0227532e+00]	 1.8887281e-01


.. parsed-literal::

      42	 1.0189390e+00	 1.7239865e-01	 1.0638698e+00	 1.8893607e-01	[ 1.0347776e+00]	 2.0323920e-01
      43	 1.0317407e+00	 1.7114811e-01	 1.0767913e+00	 1.8666521e-01	[ 1.0486557e+00]	 1.7754984e-01


.. parsed-literal::

      44	 1.0441015e+00	 1.7018937e-01	 1.0894425e+00	 1.8478997e-01	[ 1.0582205e+00]	 2.0766068e-01
      45	 1.0544632e+00	 1.6889383e-01	 1.0995881e+00	 1.8335025e-01	[ 1.0705698e+00]	 1.9582438e-01


.. parsed-literal::

      46	 1.0632701e+00	 1.6775680e-01	 1.1084362e+00	 1.8235782e-01	[ 1.0801926e+00]	 2.0749712e-01


.. parsed-literal::

      47	 1.0736611e+00	 1.6650155e-01	 1.1191811e+00	 1.8072456e-01	[ 1.0913906e+00]	 2.0664120e-01
      48	 1.0849351e+00	 1.6463992e-01	 1.1308217e+00	 1.7835396e-01	[ 1.0957974e+00]	 1.8445206e-01


.. parsed-literal::

      49	 1.0958984e+00	 1.6378031e-01	 1.1419371e+00	 1.7674973e-01	[ 1.1071023e+00]	 1.7497349e-01


.. parsed-literal::

      50	 1.1068192e+00	 1.6263202e-01	 1.1529108e+00	 1.7531439e-01	[ 1.1176582e+00]	 2.1293449e-01
      51	 1.1184516e+00	 1.6175128e-01	 1.1651110e+00	 1.7460865e-01	[ 1.1264496e+00]	 1.8993640e-01


.. parsed-literal::

      52	 1.1286905e+00	 1.6015195e-01	 1.1756111e+00	 1.7432152e-01	[ 1.1389960e+00]	 2.1035290e-01


.. parsed-literal::

      53	 1.1357435e+00	 1.5913309e-01	 1.1827443e+00	 1.7377914e-01	[ 1.1459603e+00]	 2.0498514e-01
      54	 1.1448053e+00	 1.5753063e-01	 1.1922449e+00	 1.7272753e-01	[ 1.1562671e+00]	 1.8289399e-01


.. parsed-literal::

      55	 1.1495678e+00	 1.5636620e-01	 1.1972604e+00	 1.7116915e-01	[ 1.1610364e+00]	 2.0897937e-01
      56	 1.1546172e+00	 1.5608062e-01	 1.2020941e+00	 1.7061070e-01	[ 1.1688090e+00]	 1.9897318e-01


.. parsed-literal::

      57	 1.1597219e+00	 1.5554171e-01	 1.2073882e+00	 1.6997983e-01	[ 1.1731431e+00]	 1.9894099e-01
      58	 1.1661473e+00	 1.5492911e-01	 1.2140571e+00	 1.6912251e-01	[ 1.1777568e+00]	 1.8960834e-01


.. parsed-literal::

      59	 1.1748264e+00	 1.5350680e-01	 1.2234999e+00	 1.6763827e-01	[ 1.1803588e+00]	 2.1844125e-01
      60	 1.1820410e+00	 1.5274433e-01	 1.2309172e+00	 1.6719960e-01	[ 1.1827999e+00]	 1.9896436e-01


.. parsed-literal::

      61	 1.1870552e+00	 1.5243250e-01	 1.2357358e+00	 1.6669971e-01	[ 1.1886194e+00]	 2.1033645e-01


.. parsed-literal::

      62	 1.1961655e+00	 1.5157983e-01	 1.2450249e+00	 1.6552045e-01	[ 1.1940278e+00]	 2.0387244e-01
      63	 1.2028874e+00	 1.5036291e-01	 1.2520268e+00	 1.6337452e-01	[ 1.1959183e+00]	 1.8522596e-01


.. parsed-literal::

      64	 1.2091029e+00	 1.4974907e-01	 1.2583044e+00	 1.6252288e-01	[ 1.2012427e+00]	 1.8418264e-01


.. parsed-literal::

      65	 1.2197255e+00	 1.4865578e-01	 1.2695146e+00	 1.6124321e-01	[ 1.2056351e+00]	 2.0481324e-01


.. parsed-literal::

      66	 1.2240454e+00	 1.4903506e-01	 1.2739922e+00	 1.6166069e-01	[ 1.2164587e+00]	 2.0883346e-01


.. parsed-literal::

      67	 1.2313096e+00	 1.4817442e-01	 1.2811780e+00	 1.6092814e-01	[ 1.2223832e+00]	 2.0970726e-01


.. parsed-literal::

      68	 1.2379219e+00	 1.4719174e-01	 1.2879015e+00	 1.5999218e-01	[ 1.2255256e+00]	 2.1519876e-01


.. parsed-literal::

      69	 1.2423509e+00	 1.4657162e-01	 1.2924592e+00	 1.5946691e-01	[ 1.2294276e+00]	 2.1000695e-01
      70	 1.2469482e+00	 1.4515659e-01	 1.2973287e+00	 1.5799754e-01	  1.2269834e+00 	 2.0212102e-01


.. parsed-literal::

      71	 1.2542376e+00	 1.4466217e-01	 1.3046092e+00	 1.5769241e-01	[ 1.2346440e+00]	 2.0761299e-01
      72	 1.2579513e+00	 1.4438135e-01	 1.3083638e+00	 1.5750540e-01	[ 1.2371358e+00]	 1.8610454e-01


.. parsed-literal::

      73	 1.2626495e+00	 1.4389329e-01	 1.3132924e+00	 1.5713393e-01	  1.2367723e+00 	 2.0879865e-01


.. parsed-literal::

      74	 1.2697192e+00	 1.4338076e-01	 1.3206245e+00	 1.5694261e-01	  1.2368060e+00 	 2.1198678e-01


.. parsed-literal::

      75	 1.2749927e+00	 1.4278309e-01	 1.3262247e+00	 1.5657878e-01	  1.2359006e+00 	 3.1767225e-01
      76	 1.2805548e+00	 1.4250773e-01	 1.3318499e+00	 1.5655483e-01	[ 1.2384876e+00]	 1.9874334e-01


.. parsed-literal::

      77	 1.2863390e+00	 1.4244172e-01	 1.3376258e+00	 1.5669781e-01	[ 1.2440895e+00]	 1.8139362e-01
      78	 1.2920356e+00	 1.4219199e-01	 1.3435453e+00	 1.5658719e-01	[ 1.2472438e+00]	 2.0076776e-01


.. parsed-literal::

      79	 1.2984532e+00	 1.4225370e-01	 1.3500614e+00	 1.5670293e-01	[ 1.2537189e+00]	 1.7691183e-01


.. parsed-literal::

      80	 1.3030151e+00	 1.4237143e-01	 1.3546503e+00	 1.5672757e-01	[ 1.2585628e+00]	 2.1169901e-01
      81	 1.3071113e+00	 1.4230546e-01	 1.3589416e+00	 1.5669629e-01	  1.2577697e+00 	 1.8521476e-01


.. parsed-literal::

      82	 1.3109057e+00	 1.4234791e-01	 1.3628627e+00	 1.5656647e-01	[ 1.2609110e+00]	 2.0981026e-01


.. parsed-literal::

      83	 1.3144210e+00	 1.4216545e-01	 1.3663736e+00	 1.5652053e-01	[ 1.2631616e+00]	 2.0402789e-01


.. parsed-literal::

      84	 1.3180659e+00	 1.4179777e-01	 1.3701639e+00	 1.5629922e-01	[ 1.2646169e+00]	 2.1162581e-01


.. parsed-literal::

      85	 1.3241812e+00	 1.4112231e-01	 1.3764194e+00	 1.5601351e-01	[ 1.2724145e+00]	 2.0213890e-01


.. parsed-literal::

      86	 1.3267423e+00	 1.4016879e-01	 1.3794408e+00	 1.5514379e-01	  1.2679887e+00 	 2.0286202e-01


.. parsed-literal::

      87	 1.3315670e+00	 1.4010420e-01	 1.3839854e+00	 1.5520141e-01	[ 1.2776310e+00]	 2.1029711e-01
      88	 1.3338413e+00	 1.3992374e-01	 1.3861940e+00	 1.5503101e-01	[ 1.2804865e+00]	 1.9171190e-01


.. parsed-literal::

      89	 1.3374563e+00	 1.3948148e-01	 1.3898364e+00	 1.5454231e-01	[ 1.2830114e+00]	 2.1884966e-01


.. parsed-literal::

      90	 1.3432827e+00	 1.3873845e-01	 1.3957844e+00	 1.5381234e-01	[ 1.2879107e+00]	 2.1494031e-01


.. parsed-literal::

      91	 1.3458687e+00	 1.3815847e-01	 1.3985275e+00	 1.5303980e-01	  1.2850216e+00 	 3.0733609e-01


.. parsed-literal::

      92	 1.3501082e+00	 1.3762519e-01	 1.4029234e+00	 1.5256332e-01	[ 1.2886648e+00]	 2.0885944e-01


.. parsed-literal::

      93	 1.3527778e+00	 1.3735958e-01	 1.4056670e+00	 1.5237910e-01	[ 1.2916317e+00]	 2.1461630e-01


.. parsed-literal::

      94	 1.3554854e+00	 1.3687992e-01	 1.4085312e+00	 1.5204517e-01	[ 1.2952006e+00]	 2.1267819e-01


.. parsed-literal::

      95	 1.3585175e+00	 1.3660824e-01	 1.4116655e+00	 1.5175071e-01	[ 1.2983644e+00]	 2.1863270e-01
      96	 1.3613620e+00	 1.3636476e-01	 1.4145668e+00	 1.5147300e-01	[ 1.3005254e+00]	 2.0176196e-01


.. parsed-literal::

      97	 1.3651412e+00	 1.3587940e-01	 1.4185259e+00	 1.5088697e-01	  1.3000788e+00 	 2.0106506e-01


.. parsed-literal::

      98	 1.3681562e+00	 1.3590300e-01	 1.4215096e+00	 1.5071986e-01	[ 1.3016657e+00]	 2.0536065e-01


.. parsed-literal::

      99	 1.3706496e+00	 1.3575261e-01	 1.4239426e+00	 1.5065988e-01	[ 1.3032184e+00]	 2.1258187e-01


.. parsed-literal::

     100	 1.3761171e+00	 1.3526514e-01	 1.4294118e+00	 1.5036132e-01	  1.3004248e+00 	 2.0691609e-01


.. parsed-literal::

     101	 1.3780664e+00	 1.3474835e-01	 1.4315250e+00	 1.5022015e-01	  1.2956677e+00 	 2.2082448e-01


.. parsed-literal::

     102	 1.3808691e+00	 1.3460041e-01	 1.4341958e+00	 1.5014800e-01	  1.2969359e+00 	 2.0890546e-01


.. parsed-literal::

     103	 1.3837164e+00	 1.3425722e-01	 1.4370231e+00	 1.5005687e-01	  1.2951156e+00 	 2.0912242e-01


.. parsed-literal::

     104	 1.3860130e+00	 1.3405977e-01	 1.4393115e+00	 1.5005751e-01	  1.2953804e+00 	 2.2109747e-01


.. parsed-literal::

     105	 1.3901568e+00	 1.3367178e-01	 1.4435964e+00	 1.4997585e-01	  1.2943129e+00 	 2.0404100e-01


.. parsed-literal::

     106	 1.3936280e+00	 1.3349264e-01	 1.4471523e+00	 1.5001789e-01	  1.2990488e+00 	 2.1777248e-01


.. parsed-literal::

     107	 1.3958458e+00	 1.3337639e-01	 1.4492733e+00	 1.4973965e-01	[ 1.3039542e+00]	 2.0926046e-01
     108	 1.3985720e+00	 1.3307256e-01	 1.4520411e+00	 1.4925848e-01	[ 1.3064138e+00]	 1.8731880e-01


.. parsed-literal::

     109	 1.4014592e+00	 1.3249916e-01	 1.4550323e+00	 1.4853199e-01	[ 1.3067843e+00]	 2.1018767e-01


.. parsed-literal::

     110	 1.4037919e+00	 1.3211366e-01	 1.4575328e+00	 1.4808382e-01	  1.3046462e+00 	 3.1426930e-01


.. parsed-literal::

     111	 1.4066448e+00	 1.3152516e-01	 1.4604755e+00	 1.4752636e-01	  1.3032321e+00 	 2.1016502e-01


.. parsed-literal::

     112	 1.4083660e+00	 1.3129451e-01	 1.4622092e+00	 1.4743781e-01	  1.3021725e+00 	 2.2195315e-01


.. parsed-literal::

     113	 1.4105276e+00	 1.3113201e-01	 1.4643861e+00	 1.4742539e-01	  1.3018185e+00 	 2.1946454e-01


.. parsed-literal::

     114	 1.4128873e+00	 1.3108096e-01	 1.4667849e+00	 1.4749238e-01	  1.3021662e+00 	 2.1303463e-01
     115	 1.4150739e+00	 1.3102450e-01	 1.4690067e+00	 1.4743667e-01	  1.3035657e+00 	 1.8092132e-01


.. parsed-literal::

     116	 1.4174290e+00	 1.3096407e-01	 1.4714147e+00	 1.4730483e-01	  1.3055264e+00 	 2.0680928e-01


.. parsed-literal::

     117	 1.4197986e+00	 1.3094081e-01	 1.4739429e+00	 1.4724030e-01	  1.3029293e+00 	 2.2086835e-01
     118	 1.4222514e+00	 1.3089338e-01	 1.4765179e+00	 1.4720445e-01	  1.3041045e+00 	 2.0982385e-01


.. parsed-literal::

     119	 1.4242319e+00	 1.3086185e-01	 1.4785330e+00	 1.4722873e-01	  1.3054585e+00 	 2.0283437e-01


.. parsed-literal::

     120	 1.4265849e+00	 1.3097270e-01	 1.4809402e+00	 1.4742548e-01	[ 1.3079461e+00]	 2.2556782e-01


.. parsed-literal::

     121	 1.4282561e+00	 1.3098435e-01	 1.4826431e+00	 1.4762560e-01	[ 1.3081720e+00]	 2.1320939e-01


.. parsed-literal::

     122	 1.4303230e+00	 1.3104680e-01	 1.4846656e+00	 1.4767240e-01	[ 1.3098952e+00]	 2.0633101e-01
     123	 1.4324531e+00	 1.3106674e-01	 1.4868053e+00	 1.4773777e-01	[ 1.3104852e+00]	 1.7514229e-01


.. parsed-literal::

     124	 1.4340098e+00	 1.3107473e-01	 1.4883717e+00	 1.4776539e-01	  1.3090082e+00 	 2.0306993e-01
     125	 1.4362165e+00	 1.3090080e-01	 1.4907002e+00	 1.4764825e-01	  1.3034462e+00 	 1.7805743e-01


.. parsed-literal::

     126	 1.4379131e+00	 1.3085330e-01	 1.4923803e+00	 1.4759474e-01	  1.3038213e+00 	 2.0240903e-01


.. parsed-literal::

     127	 1.4389106e+00	 1.3074700e-01	 1.4933845e+00	 1.4746708e-01	  1.3053094e+00 	 2.0975804e-01
     128	 1.4406751e+00	 1.3055649e-01	 1.4951849e+00	 1.4722693e-01	  1.3069136e+00 	 2.0014310e-01


.. parsed-literal::

     129	 1.4424014e+00	 1.3024032e-01	 1.4969910e+00	 1.4688220e-01	  1.3074744e+00 	 2.1112251e-01
     130	 1.4440452e+00	 1.3006530e-01	 1.4986801e+00	 1.4665393e-01	  1.3087174e+00 	 1.7565823e-01


.. parsed-literal::

     131	 1.4451855e+00	 1.3008899e-01	 1.4997847e+00	 1.4667221e-01	  1.3082759e+00 	 2.0913243e-01
     132	 1.4469860e+00	 1.3006086e-01	 1.5015677e+00	 1.4665079e-01	  1.3059513e+00 	 1.7641950e-01


.. parsed-literal::

     133	 1.4485170e+00	 1.2998283e-01	 1.5031558e+00	 1.4657562e-01	  1.3017562e+00 	 1.9251633e-01


.. parsed-literal::

     134	 1.4502633e+00	 1.2989688e-01	 1.5049364e+00	 1.4653155e-01	  1.2987230e+00 	 2.0556211e-01


.. parsed-literal::

     135	 1.4518581e+00	 1.2980675e-01	 1.5065869e+00	 1.4654974e-01	  1.2959466e+00 	 2.0959449e-01


.. parsed-literal::

     136	 1.4529807e+00	 1.2976138e-01	 1.5077776e+00	 1.4653299e-01	  1.2941556e+00 	 2.1595049e-01


.. parsed-literal::

     137	 1.4541247e+00	 1.2970053e-01	 1.5089751e+00	 1.4654435e-01	  1.2920766e+00 	 2.1287465e-01


.. parsed-literal::

     138	 1.4556323e+00	 1.2965588e-01	 1.5105449e+00	 1.4660647e-01	  1.2891517e+00 	 2.1002674e-01


.. parsed-literal::

     139	 1.4571812e+00	 1.2953521e-01	 1.5121655e+00	 1.4655114e-01	  1.2828023e+00 	 2.1339631e-01
     140	 1.4587172e+00	 1.2941999e-01	 1.5137189e+00	 1.4647735e-01	  1.2810481e+00 	 1.7688322e-01


.. parsed-literal::

     141	 1.4600335e+00	 1.2931578e-01	 1.5150291e+00	 1.4641213e-01	  1.2782958e+00 	 2.0153403e-01


.. parsed-literal::

     142	 1.4613493e+00	 1.2923958e-01	 1.5163283e+00	 1.4631504e-01	  1.2782356e+00 	 2.0763493e-01


.. parsed-literal::

     143	 1.4627680e+00	 1.2917180e-01	 1.5177268e+00	 1.4627057e-01	  1.2809305e+00 	 2.0825624e-01
     144	 1.4645732e+00	 1.2911035e-01	 1.5195703e+00	 1.4626859e-01	  1.2808454e+00 	 1.8508816e-01


.. parsed-literal::

     145	 1.4658872e+00	 1.2902717e-01	 1.5210122e+00	 1.4622230e-01	  1.2816385e+00 	 1.7255282e-01


.. parsed-literal::

     146	 1.4671178e+00	 1.2901501e-01	 1.5222349e+00	 1.4622357e-01	  1.2831660e+00 	 2.1552253e-01
     147	 1.4680976e+00	 1.2900681e-01	 1.5232415e+00	 1.4623677e-01	  1.2837863e+00 	 1.9836783e-01


.. parsed-literal::

     148	 1.4693464e+00	 1.2902150e-01	 1.5245320e+00	 1.4627061e-01	  1.2840715e+00 	 2.0543647e-01


.. parsed-literal::

     149	 1.4704237e+00	 1.2906042e-01	 1.5257323e+00	 1.4644032e-01	  1.2842486e+00 	 2.1978545e-01


.. parsed-literal::

     150	 1.4716530e+00	 1.2910097e-01	 1.5269061e+00	 1.4649185e-01	  1.2843250e+00 	 2.1719122e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.14 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdbbf237910>



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
    CPU times: user 1.9 s, sys: 38 ms, total: 1.94 s
    Wall time: 662 ms


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

