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
       1	-3.2695753e-01	 3.1527721e-01	-3.1709176e-01	 3.4073031e-01	[-3.6785523e-01]	 4.7312737e-01


.. parsed-literal::

       2	-2.5086825e-01	 3.0247841e-01	-2.2511329e-01	 3.2894648e-01	[-3.0695790e-01]	 2.3557854e-01


.. parsed-literal::

       3	-2.0820961e-01	 2.8418009e-01	-1.6734280e-01	 3.1021854e-01	[-2.7852824e-01]	 2.8044295e-01


.. parsed-literal::

       4	-1.6641973e-01	 2.6624520e-01	-1.1751739e-01	 2.9018245e-01	[-2.3993105e-01]	 2.9296398e-01


.. parsed-literal::

       5	-1.3235875e-01	 2.5458522e-01	-1.0288531e-01	 2.7430086e-01	[-2.1561871e-01]	 2.1626139e-01


.. parsed-literal::

       6	-6.5388256e-02	 2.5094482e-01	-3.6472616e-02	 2.6689298e-01	[-1.0480889e-01]	 2.0138288e-01


.. parsed-literal::

       7	-4.6481397e-02	 2.4706575e-01	-2.2219014e-02	 2.6388562e-01	[-8.4624842e-02]	 2.0324993e-01


.. parsed-literal::

       8	-3.5502115e-02	 2.4517101e-01	-1.3824326e-02	 2.6113098e-01	[-7.6237483e-02]	 2.0182419e-01


.. parsed-literal::

       9	-1.9404206e-02	 2.4231716e-01	-1.3381399e-03	 2.5832863e-01	[-6.7855343e-02]	 2.0804477e-01
      10	-7.2024222e-03	 2.3997514e-01	 8.9350591e-03	 2.5579012e-01	[-5.9454905e-02]	 1.7819738e-01


.. parsed-literal::

      11	 9.3082452e-05	 2.3880854e-01	 1.4699603e-02	 2.5522084e-01	[-5.6807713e-02]	 1.9424558e-01
      12	 5.4594105e-03	 2.3773315e-01	 1.9569123e-02	 2.5354121e-01	[-4.7912716e-02]	 1.9919825e-01


.. parsed-literal::

      13	 8.0923780e-03	 2.3718509e-01	 2.2150050e-02	 2.5325380e-01	[-4.5977963e-02]	 2.1689200e-01


.. parsed-literal::

      14	 1.7454857e-02	 2.3499388e-01	 3.3265644e-02	 2.5241708e-01	[-4.1412433e-02]	 2.1349525e-01


.. parsed-literal::

      15	 1.0963080e-01	 2.2122960e-01	 1.3267203e-01	 2.4505215e-01	[ 5.3196206e-02]	 2.1245527e-01


.. parsed-literal::

      16	 1.5654023e-01	 2.2191585e-01	 1.8122755e-01	 2.4616545e-01	[ 1.2762192e-01]	 3.2926393e-01


.. parsed-literal::

      17	 2.5030983e-01	 2.1701702e-01	 2.7717978e-01	 2.3435603e-01	[ 2.1660994e-01]	 2.1113253e-01


.. parsed-literal::

      18	 3.0031087e-01	 2.1326456e-01	 3.3182381e-01	 2.3113780e-01	[ 2.4857371e-01]	 2.2206140e-01


.. parsed-literal::

      19	 3.3823246e-01	 2.0984538e-01	 3.7236419e-01	 2.2705134e-01	[ 2.9144763e-01]	 2.2117543e-01


.. parsed-literal::

      20	 3.8613396e-01	 2.0688264e-01	 4.1993195e-01	 2.2354161e-01	[ 3.4818313e-01]	 2.2090149e-01


.. parsed-literal::

      21	 4.2892816e-01	 2.0548767e-01	 4.6191363e-01	 2.2279646e-01	[ 4.0099656e-01]	 2.1432734e-01


.. parsed-literal::

      22	 4.9391206e-01	 2.0266361e-01	 5.2803928e-01	 2.2164938e-01	[ 4.7131214e-01]	 2.1608305e-01
      23	 5.7475319e-01	 2.0359475e-01	 6.1458318e-01	 2.2338361e-01	[ 5.8459828e-01]	 1.7863035e-01


.. parsed-literal::

      24	 6.0651525e-01	 2.0397494e-01	 6.4455292e-01	 2.2034483e-01	[ 6.3845253e-01]	 2.0432115e-01


.. parsed-literal::

      25	 6.3659841e-01	 2.0048807e-01	 6.7394712e-01	 2.1621570e-01	[ 6.6416637e-01]	 2.1247220e-01


.. parsed-literal::

      26	 6.3817233e-01	 2.0645301e-01	 6.7483852e-01	 2.2636037e-01	  6.6006886e-01 	 2.1864104e-01


.. parsed-literal::

      27	 6.8520548e-01	 2.0247568e-01	 7.2160582e-01	 2.2467013e-01	[ 7.0857452e-01]	 2.1038008e-01


.. parsed-literal::

      28	 7.0687624e-01	 2.0054141e-01	 7.4415937e-01	 2.2145373e-01	[ 7.3548152e-01]	 2.1608877e-01
      29	 7.5801904e-01	 2.0177697e-01	 7.9612486e-01	 2.2263418e-01	[ 8.0125351e-01]	 1.8419027e-01


.. parsed-literal::

      30	 7.8160770e-01	 2.0785948e-01	 8.2129167e-01	 2.2991062e-01	[ 8.2347380e-01]	 2.0607352e-01


.. parsed-literal::

      31	 8.1254025e-01	 2.0562384e-01	 8.5312544e-01	 2.2717819e-01	[ 8.5558188e-01]	 2.1116090e-01
      32	 8.2912581e-01	 2.0432871e-01	 8.6971094e-01	 2.2729566e-01	[ 8.6998047e-01]	 1.8234491e-01


.. parsed-literal::

      33	 8.4855290e-01	 2.0284692e-01	 8.9036548e-01	 2.2612022e-01	[ 8.8719688e-01]	 2.0055389e-01
      34	 8.7513733e-01	 2.0500457e-01	 9.1708832e-01	 2.2864215e-01	[ 9.0428939e-01]	 1.9403434e-01


.. parsed-literal::

      35	 8.9238464e-01	 2.0209046e-01	 9.3602527e-01	 2.2363871e-01	[ 9.1867226e-01]	 2.0908904e-01


.. parsed-literal::

      36	 9.1561219e-01	 2.0164356e-01	 9.5864020e-01	 2.2405678e-01	[ 9.3217854e-01]	 2.1168876e-01
      37	 9.3015926e-01	 1.9979652e-01	 9.7323801e-01	 2.2168421e-01	[ 9.4247946e-01]	 1.8342137e-01


.. parsed-literal::

      38	 9.5687622e-01	 1.9447167e-01	 1.0009018e+00	 2.1455638e-01	[ 9.6107062e-01]	 2.1644878e-01


.. parsed-literal::

      39	 9.7857090e-01	 1.8947152e-01	 1.0243368e+00	 2.0789207e-01	[ 9.8862390e-01]	 2.1537352e-01
      40	 9.9440762e-01	 1.8589620e-01	 1.0404550e+00	 2.0269477e-01	[ 9.9480762e-01]	 1.8815112e-01


.. parsed-literal::

      41	 1.0059235e+00	 1.8449035e-01	 1.0519824e+00	 2.0121928e-01	[ 1.0018515e+00]	 2.0448017e-01


.. parsed-literal::

      42	 1.0278144e+00	 1.7943733e-01	 1.0750823e+00	 1.9537938e-01	[ 1.0085244e+00]	 2.1474004e-01


.. parsed-literal::

      43	 1.0410210e+00	 1.7751160e-01	 1.0885237e+00	 1.9214899e-01	[ 1.0134382e+00]	 2.0840168e-01


.. parsed-literal::

      44	 1.0531697e+00	 1.7538056e-01	 1.1005662e+00	 1.8952225e-01	[ 1.0215026e+00]	 2.0669699e-01


.. parsed-literal::

      45	 1.0699868e+00	 1.7198519e-01	 1.1175385e+00	 1.8517108e-01	[ 1.0295885e+00]	 2.1058345e-01


.. parsed-literal::

      46	 1.0924721e+00	 1.6832760e-01	 1.1402820e+00	 1.8047170e-01	[ 1.0360939e+00]	 2.0932126e-01


.. parsed-literal::

      47	 1.1099290e+00	 1.6360178e-01	 1.1583723e+00	 1.7516250e-01	[ 1.0524278e+00]	 2.1569920e-01


.. parsed-literal::

      48	 1.1254886e+00	 1.6185645e-01	 1.1741519e+00	 1.7280961e-01	[ 1.0750994e+00]	 2.0444202e-01


.. parsed-literal::

      49	 1.1408827e+00	 1.5933540e-01	 1.1900835e+00	 1.7009156e-01	[ 1.0916627e+00]	 2.1005678e-01


.. parsed-literal::

      50	 1.1583647e+00	 1.5579748e-01	 1.2081525e+00	 1.6594168e-01	[ 1.1066980e+00]	 2.2022390e-01


.. parsed-literal::

      51	 1.1715060e+00	 1.5262553e-01	 1.2218418e+00	 1.6275620e-01	[ 1.1182019e+00]	 2.1291518e-01
      52	 1.1825197e+00	 1.5147923e-01	 1.2326245e+00	 1.6270545e-01	[ 1.1185400e+00]	 1.9525385e-01


.. parsed-literal::

      53	 1.1923992e+00	 1.5035669e-01	 1.2428272e+00	 1.6223824e-01	  1.1162343e+00 	 2.0429111e-01


.. parsed-literal::

      54	 1.2041638e+00	 1.4933639e-01	 1.2546878e+00	 1.6121988e-01	[ 1.1249802e+00]	 2.1118307e-01


.. parsed-literal::

      55	 1.2126518e+00	 1.4621711e-01	 1.2639592e+00	 1.5760090e-01	[ 1.1279324e+00]	 2.1942568e-01


.. parsed-literal::

      56	 1.2299964e+00	 1.4631028e-01	 1.2807228e+00	 1.5731609e-01	[ 1.1525157e+00]	 2.0726085e-01


.. parsed-literal::

      57	 1.2376392e+00	 1.4590232e-01	 1.2884112e+00	 1.5631381e-01	[ 1.1644919e+00]	 2.0857430e-01


.. parsed-literal::

      58	 1.2538687e+00	 1.4516811e-01	 1.3048874e+00	 1.5420039e-01	[ 1.1865379e+00]	 2.0467210e-01
      59	 1.2678401e+00	 1.4419708e-01	 1.3190758e+00	 1.5205386e-01	[ 1.1983892e+00]	 1.9696689e-01


.. parsed-literal::

      60	 1.2795451e+00	 1.4298133e-01	 1.3307107e+00	 1.5090227e-01	[ 1.2108104e+00]	 1.8299246e-01


.. parsed-literal::

      61	 1.2888415e+00	 1.4161972e-01	 1.3401838e+00	 1.5008517e-01	[ 1.2165700e+00]	 2.1526265e-01


.. parsed-literal::

      62	 1.3000970e+00	 1.3939197e-01	 1.3517960e+00	 1.4927338e-01	  1.2148969e+00 	 2.2200441e-01


.. parsed-literal::

      63	 1.3078000e+00	 1.3902011e-01	 1.3596613e+00	 1.4914572e-01	[ 1.2238406e+00]	 2.1525192e-01


.. parsed-literal::

      64	 1.3159202e+00	 1.3918778e-01	 1.3677954e+00	 1.4916439e-01	[ 1.2306838e+00]	 2.0656610e-01


.. parsed-literal::

      65	 1.3279725e+00	 1.3911198e-01	 1.3803433e+00	 1.4997321e-01	[ 1.2353582e+00]	 2.1448588e-01


.. parsed-literal::

      66	 1.3337659e+00	 1.3848740e-01	 1.3862737e+00	 1.5057948e-01	  1.2341568e+00 	 2.1324801e-01
      67	 1.3407331e+00	 1.3771319e-01	 1.3931475e+00	 1.5033296e-01	[ 1.2407242e+00]	 1.9041610e-01


.. parsed-literal::

      68	 1.3467318e+00	 1.3644848e-01	 1.3992492e+00	 1.4962602e-01	[ 1.2413747e+00]	 2.1071625e-01
      69	 1.3531602e+00	 1.3598315e-01	 1.4059026e+00	 1.4951466e-01	  1.2410230e+00 	 1.9087148e-01


.. parsed-literal::

      70	 1.3605953e+00	 1.3458973e-01	 1.4133590e+00	 1.4849545e-01	  1.2404637e+00 	 2.1080184e-01


.. parsed-literal::

      71	 1.3670498e+00	 1.3542462e-01	 1.4198650e+00	 1.4919444e-01	[ 1.2439437e+00]	 2.1290135e-01


.. parsed-literal::

      72	 1.3721467e+00	 1.3546183e-01	 1.4249522e+00	 1.4924654e-01	[ 1.2517566e+00]	 2.1482515e-01


.. parsed-literal::

      73	 1.3777437e+00	 1.3460565e-01	 1.4305957e+00	 1.4914059e-01	[ 1.2595479e+00]	 2.1591306e-01


.. parsed-literal::

      74	 1.3841010e+00	 1.3407799e-01	 1.4369687e+00	 1.4932286e-01	[ 1.2705838e+00]	 2.1801925e-01


.. parsed-literal::

      75	 1.3896049e+00	 1.3341225e-01	 1.4424137e+00	 1.4911802e-01	[ 1.2790235e+00]	 2.1104383e-01
      76	 1.3952457e+00	 1.3326848e-01	 1.4482476e+00	 1.4943758e-01	[ 1.2829787e+00]	 1.9313002e-01


.. parsed-literal::

      77	 1.3995028e+00	 1.3310179e-01	 1.4526013e+00	 1.4843381e-01	[ 1.2836982e+00]	 2.1389556e-01


.. parsed-literal::

      78	 1.4034593e+00	 1.3354647e-01	 1.4565693e+00	 1.4818289e-01	[ 1.2855416e+00]	 2.1586657e-01


.. parsed-literal::

      79	 1.4083718e+00	 1.3427577e-01	 1.4616587e+00	 1.4782469e-01	  1.2851565e+00 	 2.0653820e-01
      80	 1.4124402e+00	 1.3470915e-01	 1.4658790e+00	 1.4734438e-01	[ 1.2860664e+00]	 1.8111610e-01


.. parsed-literal::

      81	 1.4169087e+00	 1.3455084e-01	 1.4703435e+00	 1.4714229e-01	[ 1.2875609e+00]	 2.1385908e-01
      82	 1.4209520e+00	 1.3415309e-01	 1.4744543e+00	 1.4707672e-01	[ 1.2899026e+00]	 1.7981005e-01


.. parsed-literal::

      83	 1.4247087e+00	 1.3365833e-01	 1.4783449e+00	 1.4673974e-01	[ 1.2934770e+00]	 2.0283937e-01
      84	 1.4283159e+00	 1.3332573e-01	 1.4821116e+00	 1.4614851e-01	[ 1.2946154e+00]	 1.9018579e-01


.. parsed-literal::

      85	 1.4312739e+00	 1.3319232e-01	 1.4850557e+00	 1.4563788e-01	[ 1.2962345e+00]	 2.1529031e-01


.. parsed-literal::

      86	 1.4346057e+00	 1.3310441e-01	 1.4884367e+00	 1.4483829e-01	  1.2957695e+00 	 2.1036530e-01


.. parsed-literal::

      87	 1.4381730e+00	 1.3273234e-01	 1.4920722e+00	 1.4432794e-01	  1.2953661e+00 	 2.1138954e-01


.. parsed-literal::

      88	 1.4420408e+00	 1.3230967e-01	 1.4959698e+00	 1.4394900e-01	  1.2941602e+00 	 2.1671486e-01


.. parsed-literal::

      89	 1.4447468e+00	 1.3181407e-01	 1.4986174e+00	 1.4370698e-01	  1.2961868e+00 	 2.1260953e-01


.. parsed-literal::

      90	 1.4492303e+00	 1.3114176e-01	 1.5030030e+00	 1.4362757e-01	[ 1.2965302e+00]	 2.0244765e-01


.. parsed-literal::

      91	 1.4500785e+00	 1.3009878e-01	 1.5039100e+00	 1.4270348e-01	[ 1.2999695e+00]	 2.0959902e-01


.. parsed-literal::

      92	 1.4544685e+00	 1.3031374e-01	 1.5081779e+00	 1.4284971e-01	[ 1.3006558e+00]	 2.0676255e-01


.. parsed-literal::

      93	 1.4563761e+00	 1.3031932e-01	 1.5101049e+00	 1.4274383e-01	[ 1.3011032e+00]	 2.1266437e-01


.. parsed-literal::

      94	 1.4591203e+00	 1.3018602e-01	 1.5129290e+00	 1.4248160e-01	  1.3006632e+00 	 2.0497489e-01


.. parsed-literal::

      95	 1.4626792e+00	 1.3004380e-01	 1.5165695e+00	 1.4205440e-01	[ 1.3027494e+00]	 2.0599842e-01


.. parsed-literal::

      96	 1.4635669e+00	 1.2990913e-01	 1.5177331e+00	 1.4144083e-01	  1.2929149e+00 	 2.1349645e-01


.. parsed-literal::

      97	 1.4672537e+00	 1.2981937e-01	 1.5212497e+00	 1.4129162e-01	  1.3011622e+00 	 2.0524597e-01
      98	 1.4683381e+00	 1.2980775e-01	 1.5223279e+00	 1.4122870e-01	[ 1.3027797e+00]	 2.0513678e-01


.. parsed-literal::

      99	 1.4705022e+00	 1.2984309e-01	 1.5246334e+00	 1.4087281e-01	  1.3002767e+00 	 2.0440674e-01
     100	 1.4726700e+00	 1.2991157e-01	 1.5269159e+00	 1.4082565e-01	  1.2974585e+00 	 1.9792628e-01


.. parsed-literal::

     101	 1.4746795e+00	 1.2991044e-01	 1.5289839e+00	 1.4075852e-01	  1.2958547e+00 	 1.9252491e-01


.. parsed-literal::

     102	 1.4777082e+00	 1.2990351e-01	 1.5322060e+00	 1.4073776e-01	  1.2887662e+00 	 2.1208119e-01


.. parsed-literal::

     103	 1.4796024e+00	 1.2980020e-01	 1.5341772e+00	 1.4078056e-01	  1.2863725e+00 	 2.1232677e-01


.. parsed-literal::

     104	 1.4819516e+00	 1.2976136e-01	 1.5365090e+00	 1.4083760e-01	  1.2843946e+00 	 2.1022677e-01


.. parsed-literal::

     105	 1.4848114e+00	 1.2984290e-01	 1.5394123e+00	 1.4088978e-01	  1.2775068e+00 	 2.0894504e-01


.. parsed-literal::

     106	 1.4867769e+00	 1.2997555e-01	 1.5413881e+00	 1.4085017e-01	  1.2768104e+00 	 2.1024442e-01
     107	 1.4892948e+00	 1.3023045e-01	 1.5439070e+00	 1.4086531e-01	  1.2735696e+00 	 1.8318534e-01


.. parsed-literal::

     108	 1.4908684e+00	 1.3041794e-01	 1.5455802e+00	 1.4081861e-01	  1.2673754e+00 	 1.9818068e-01


.. parsed-literal::

     109	 1.4927682e+00	 1.3047180e-01	 1.5474264e+00	 1.4078310e-01	  1.2726283e+00 	 2.1461344e-01


.. parsed-literal::

     110	 1.4940453e+00	 1.3047497e-01	 1.5487531e+00	 1.4078523e-01	  1.2719259e+00 	 2.1004462e-01


.. parsed-literal::

     111	 1.4959753e+00	 1.3047892e-01	 1.5508109e+00	 1.4068396e-01	  1.2682967e+00 	 2.1108079e-01
     112	 1.4971153e+00	 1.3052008e-01	 1.5521253e+00	 1.4082549e-01	  1.2622385e+00 	 1.8997455e-01


.. parsed-literal::

     113	 1.4987031e+00	 1.3053889e-01	 1.5536568e+00	 1.4071785e-01	  1.2620495e+00 	 1.9675636e-01
     114	 1.4999910e+00	 1.3053444e-01	 1.5549195e+00	 1.4064770e-01	  1.2616674e+00 	 1.8320203e-01


.. parsed-literal::

     115	 1.5015469e+00	 1.3049177e-01	 1.5565183e+00	 1.4061038e-01	  1.2586697e+00 	 1.7806458e-01


.. parsed-literal::

     116	 1.5044824e+00	 1.3041922e-01	 1.5596173e+00	 1.4066224e-01	  1.2487568e+00 	 2.0829368e-01


.. parsed-literal::

     117	 1.5059946e+00	 1.3020423e-01	 1.5612638e+00	 1.4057728e-01	  1.2405373e+00 	 3.0177593e-01
     118	 1.5081780e+00	 1.3012631e-01	 1.5635606e+00	 1.4067476e-01	  1.2301502e+00 	 1.7961597e-01


.. parsed-literal::

     119	 1.5098650e+00	 1.2999139e-01	 1.5653093e+00	 1.4072628e-01	  1.2239210e+00 	 2.1103358e-01


.. parsed-literal::

     120	 1.5115947e+00	 1.2986247e-01	 1.5671103e+00	 1.4077276e-01	  1.2139032e+00 	 2.1250820e-01


.. parsed-literal::

     121	 1.5133162e+00	 1.2974503e-01	 1.5687480e+00	 1.4079483e-01	  1.2154588e+00 	 2.1594453e-01


.. parsed-literal::

     122	 1.5148519e+00	 1.2969166e-01	 1.5701785e+00	 1.4077806e-01	  1.2181783e+00 	 2.0650363e-01


.. parsed-literal::

     123	 1.5162530e+00	 1.2962298e-01	 1.5715373e+00	 1.4082024e-01	  1.2153388e+00 	 2.1925926e-01


.. parsed-literal::

     124	 1.5181218e+00	 1.2959509e-01	 1.5733780e+00	 1.4090005e-01	  1.2040142e+00 	 2.1077371e-01
     125	 1.5197641e+00	 1.2949469e-01	 1.5750413e+00	 1.4106324e-01	  1.1882867e+00 	 1.9165039e-01


.. parsed-literal::

     126	 1.5216322e+00	 1.2935512e-01	 1.5769915e+00	 1.4108049e-01	  1.1675385e+00 	 2.1699095e-01


.. parsed-literal::

     127	 1.5233805e+00	 1.2928488e-01	 1.5787992e+00	 1.4103194e-01	  1.1471540e+00 	 2.1521688e-01


.. parsed-literal::

     128	 1.5248478e+00	 1.2906350e-01	 1.5803354e+00	 1.4084341e-01	  1.1355531e+00 	 2.1608090e-01


.. parsed-literal::

     129	 1.5259251e+00	 1.2912713e-01	 1.5813931e+00	 1.4068896e-01	  1.1341684e+00 	 2.1475315e-01


.. parsed-literal::

     130	 1.5272323e+00	 1.2910169e-01	 1.5826731e+00	 1.4062112e-01	  1.1308784e+00 	 2.0413136e-01
     131	 1.5291993e+00	 1.2904929e-01	 1.5846745e+00	 1.4066734e-01	  1.1159800e+00 	 1.8632245e-01


.. parsed-literal::

     132	 1.5306330e+00	 1.2882706e-01	 1.5862104e+00	 1.4070314e-01	  1.0948495e+00 	 1.8687010e-01


.. parsed-literal::

     133	 1.5319410e+00	 1.2876406e-01	 1.5874967e+00	 1.4074937e-01	  1.0896103e+00 	 2.0748448e-01


.. parsed-literal::

     134	 1.5335290e+00	 1.2855725e-01	 1.5891337e+00	 1.4085935e-01	  1.0694335e+00 	 2.0596409e-01


.. parsed-literal::

     135	 1.5350936e+00	 1.2834842e-01	 1.5907679e+00	 1.4093092e-01	  1.0454460e+00 	 2.1409607e-01
     136	 1.5365895e+00	 1.2776988e-01	 1.5924788e+00	 1.4095675e-01	  9.9164807e-01 	 1.9324660e-01


.. parsed-literal::

     137	 1.5383936e+00	 1.2777666e-01	 1.5942341e+00	 1.4093885e-01	  9.8934397e-01 	 2.1423197e-01


.. parsed-literal::

     138	 1.5393259e+00	 1.2772453e-01	 1.5951334e+00	 1.4087266e-01	  9.9002587e-01 	 2.2276139e-01


.. parsed-literal::

     139	 1.5409475e+00	 1.2753014e-01	 1.5967660e+00	 1.4085607e-01	  9.7180996e-01 	 2.1486902e-01


.. parsed-literal::

     140	 1.5419071e+00	 1.2715689e-01	 1.5979377e+00	 1.4103210e-01	  9.3469043e-01 	 2.1161652e-01


.. parsed-literal::

     141	 1.5437474e+00	 1.2713998e-01	 1.5996201e+00	 1.4108476e-01	  9.2202733e-01 	 2.1173596e-01
     142	 1.5446196e+00	 1.2705315e-01	 1.6005151e+00	 1.4118360e-01	  9.0285053e-01 	 1.9240212e-01


.. parsed-literal::

     143	 1.5457606e+00	 1.2696853e-01	 1.6017002e+00	 1.4134378e-01	  8.7218369e-01 	 2.1482134e-01


.. parsed-literal::

     144	 1.5471838e+00	 1.2677145e-01	 1.6032137e+00	 1.4138952e-01	  8.3491438e-01 	 2.1404600e-01


.. parsed-literal::

     145	 1.5487979e+00	 1.2681304e-01	 1.6048659e+00	 1.4163051e-01	  7.8760656e-01 	 2.1147156e-01


.. parsed-literal::

     146	 1.5500622e+00	 1.2684451e-01	 1.6061351e+00	 1.4151719e-01	  7.7370888e-01 	 2.1440911e-01
     147	 1.5511036e+00	 1.2679851e-01	 1.6072600e+00	 1.4138487e-01	  7.5327229e-01 	 1.9984341e-01


.. parsed-literal::

     148	 1.5520512e+00	 1.2679262e-01	 1.6081875e+00	 1.4122219e-01	  7.5165730e-01 	 2.2176123e-01


.. parsed-literal::

     149	 1.5529393e+00	 1.2675302e-01	 1.6090641e+00	 1.4118824e-01	  7.4353331e-01 	 2.1249557e-01


.. parsed-literal::

     150	 1.5544864e+00	 1.2668034e-01	 1.6106295e+00	 1.4131605e-01	  7.1495847e-01 	 2.0941758e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.17 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f27a41af6a0>



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
    CPU times: user 2.2 s, sys: 57 ms, total: 2.26 s
    Wall time: 723 ms


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

