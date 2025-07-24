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
       1	-3.5117132e-01	 3.2281098e-01	-3.4156509e-01	 3.1116866e-01	[-3.2039388e-01]	 4.5896649e-01


.. parsed-literal::

       2	-2.8073693e-01	 3.1258107e-01	-2.5719405e-01	 3.0185173e-01	[-2.2536057e-01]	 2.3027182e-01


.. parsed-literal::

       3	-2.3544414e-01	 2.9080077e-01	-1.9254051e-01	 2.8122150e-01	[-1.5473502e-01]	 2.8604031e-01


.. parsed-literal::

       4	-1.9617485e-01	 2.6693544e-01	-1.5378551e-01	 2.6014672e-01	[-1.1693504e-01]	 2.1830130e-01


.. parsed-literal::

       5	-1.0562840e-01	 2.5777053e-01	-7.2880525e-02	 2.5190885e-01	[-4.7448999e-02]	 2.1325231e-01
       6	-7.2829310e-02	 2.5247200e-01	-4.3716858e-02	 2.4807361e-01	[-2.8376063e-02]	 1.8773675e-01


.. parsed-literal::

       7	-5.5440989e-02	 2.4975182e-01	-3.1980533e-02	 2.4547102e-01	[-1.5566562e-02]	 1.8213487e-01


.. parsed-literal::

       8	-4.2296002e-02	 2.4748859e-01	-2.2551720e-02	 2.4312491e-01	[-5.3597574e-03]	 2.0501161e-01


.. parsed-literal::

       9	-2.9378370e-02	 2.4504128e-01	-1.2166906e-02	 2.4098287e-01	[ 3.9705067e-03]	 2.0366287e-01


.. parsed-literal::

      10	-1.7583424e-02	 2.4276057e-01	-2.2487941e-03	 2.3966184e-01	[ 1.0263939e-02]	 2.1379280e-01


.. parsed-literal::

      11	-1.4264483e-02	 2.4245302e-01	-5.9707197e-05	 2.3933566e-01	[ 1.2120886e-02]	 2.0529032e-01


.. parsed-literal::

      12	-1.0993279e-02	 2.4167852e-01	 2.9906254e-03	 2.3878250e-01	[ 1.4352805e-02]	 2.0863390e-01


.. parsed-literal::

      13	-7.9470605e-03	 2.4104323e-01	 6.0232901e-03	 2.3832917e-01	[ 1.6926795e-02]	 2.0877266e-01
      14	-3.7257727e-03	 2.4012154e-01	 1.0679011e-02	 2.3752743e-01	[ 2.0764253e-02]	 1.8477201e-01


.. parsed-literal::

      15	 9.0519211e-02	 2.2632548e-01	 1.1023439e-01	 2.2658164e-01	[ 1.0771327e-01]	 3.1437469e-01
      16	 1.7847800e-01	 2.2136319e-01	 2.0165044e-01	 2.2324107e-01	[ 1.9605767e-01]	 1.9338632e-01


.. parsed-literal::

      17	 2.6962903e-01	 2.1843714e-01	 3.0138035e-01	 2.1865794e-01	[ 3.0665008e-01]	 2.1562004e-01


.. parsed-literal::

      18	 3.0906028e-01	 2.1974018e-01	 3.3977530e-01	 2.1884818e-01	[ 3.4835998e-01]	 2.1378088e-01


.. parsed-literal::

      19	 3.4443506e-01	 2.1556453e-01	 3.7557709e-01	 2.1665509e-01	[ 3.7980890e-01]	 2.1212149e-01
      20	 3.8985893e-01	 2.1025174e-01	 4.2251986e-01	 2.1194283e-01	[ 4.2365576e-01]	 2.0009995e-01


.. parsed-literal::

      21	 4.5669791e-01	 2.0909674e-01	 4.9113399e-01	 2.0873604e-01	[ 4.9144096e-01]	 2.1524644e-01


.. parsed-literal::

      22	 5.4632422e-01	 2.0721783e-01	 5.8199985e-01	 2.0715472e-01	[ 5.8304056e-01]	 2.1400905e-01
      23	 5.9331087e-01	 2.0311837e-01	 6.3265594e-01	 2.0354925e-01	[ 6.3208911e-01]	 1.9810510e-01


.. parsed-literal::

      24	 6.3066566e-01	 2.0118020e-01	 6.6814650e-01	 2.0011879e-01	[ 6.8124819e-01]	 2.0220709e-01
      25	 6.6051709e-01	 1.9817137e-01	 6.9806775e-01	 1.9712555e-01	[ 7.0940935e-01]	 1.9151640e-01


.. parsed-literal::

      26	 6.7955106e-01	 2.0376185e-01	 7.1563223e-01	 2.0193139e-01	[ 7.1882497e-01]	 2.0160651e-01


.. parsed-literal::

      27	 7.1155630e-01	 2.0471959e-01	 7.4732209e-01	 2.0500907e-01	[ 7.4492186e-01]	 2.1411896e-01
      28	 7.2720825e-01	 2.0084863e-01	 7.6432000e-01	 2.0028107e-01	[ 7.7599045e-01]	 1.9738960e-01


.. parsed-literal::

      29	 7.5893942e-01	 1.9763057e-01	 7.9675879e-01	 1.9623176e-01	[ 8.2019562e-01]	 2.0352197e-01
      30	 7.9139746e-01	 1.9668622e-01	 8.2959337e-01	 1.9461375e-01	[ 8.5743412e-01]	 1.9973159e-01


.. parsed-literal::

      31	 8.3501406e-01	 1.9612572e-01	 8.7420675e-01	 1.9298746e-01	[ 8.9250853e-01]	 2.0669198e-01


.. parsed-literal::

      32	 8.5769637e-01	 1.9156440e-01	 8.9871557e-01	 1.8871394e-01	[ 9.2367732e-01]	 2.1645379e-01
      33	 8.7864152e-01	 1.9078917e-01	 9.1989145e-01	 1.8682161e-01	[ 9.4074376e-01]	 1.9956255e-01


.. parsed-literal::

      34	 8.9235152e-01	 1.8909273e-01	 9.3391569e-01	 1.8575693e-01	[ 9.5005829e-01]	 1.8280482e-01


.. parsed-literal::

      35	 9.0988372e-01	 1.8652272e-01	 9.5187415e-01	 1.8403362e-01	[ 9.6193762e-01]	 2.0548773e-01


.. parsed-literal::

      36	 9.3145867e-01	 1.8296182e-01	 9.7439425e-01	 1.8129798e-01	[ 9.6614441e-01]	 2.2066092e-01


.. parsed-literal::

      37	 9.4820695e-01	 1.8100621e-01	 9.9140745e-01	 1.7907550e-01	[ 9.8907877e-01]	 2.1492696e-01


.. parsed-literal::

      38	 9.5668561e-01	 1.8010295e-01	 9.9982153e-01	 1.7808536e-01	[ 9.9684710e-01]	 2.0530200e-01


.. parsed-literal::

      39	 9.7614675e-01	 1.7833199e-01	 1.0196711e+00	 1.7622478e-01	[ 1.0053786e+00]	 2.0737219e-01


.. parsed-literal::

      40	 9.8601289e-01	 1.7854847e-01	 1.0301246e+00	 1.7702547e-01	[ 1.0079322e+00]	 2.0581079e-01


.. parsed-literal::

      41	 1.0008438e+00	 1.7769231e-01	 1.0451148e+00	 1.7604447e-01	[ 1.0159593e+00]	 2.1656680e-01


.. parsed-literal::

      42	 1.0129363e+00	 1.7745578e-01	 1.0580830e+00	 1.7542205e-01	[ 1.0199862e+00]	 2.1203160e-01
      43	 1.0231865e+00	 1.7735112e-01	 1.0690883e+00	 1.7511109e-01	[ 1.0279755e+00]	 1.8431568e-01


.. parsed-literal::

      44	 1.0383784e+00	 1.7793288e-01	 1.0857988e+00	 1.7477367e-01	[ 1.0529408e+00]	 2.0584941e-01
      45	 1.0489626e+00	 1.7726895e-01	 1.0968121e+00	 1.7368375e-01	[ 1.0782536e+00]	 1.7280245e-01


.. parsed-literal::

      46	 1.0578747e+00	 1.7594136e-01	 1.1050151e+00	 1.7204894e-01	[ 1.0860842e+00]	 1.7667341e-01


.. parsed-literal::

      47	 1.0674393e+00	 1.7395433e-01	 1.1144051e+00	 1.7009866e-01	[ 1.0948053e+00]	 2.0686269e-01


.. parsed-literal::

      48	 1.0778659e+00	 1.7186077e-01	 1.1248946e+00	 1.6774910e-01	[ 1.1069814e+00]	 2.1115518e-01


.. parsed-literal::

      49	 1.0914027e+00	 1.6852920e-01	 1.1386139e+00	 1.6523363e-01	[ 1.1198462e+00]	 2.0418262e-01


.. parsed-literal::

      50	 1.1011100e+00	 1.6628149e-01	 1.1484414e+00	 1.6311215e-01	[ 1.1220434e+00]	 2.0380378e-01


.. parsed-literal::

      51	 1.1083944e+00	 1.6544293e-01	 1.1557330e+00	 1.6226499e-01	[ 1.1269628e+00]	 2.0524573e-01
      52	 1.1172383e+00	 1.6418313e-01	 1.1650425e+00	 1.6117634e-01	[ 1.1346184e+00]	 1.8335581e-01


.. parsed-literal::

      53	 1.1280447e+00	 1.6240659e-01	 1.1764122e+00	 1.6038134e-01	[ 1.1390467e+00]	 1.8282390e-01
      54	 1.1374796e+00	 1.6029068e-01	 1.1866821e+00	 1.6044913e-01	[ 1.1413082e+00]	 1.9693995e-01


.. parsed-literal::

      55	 1.1469107e+00	 1.5957538e-01	 1.1958233e+00	 1.5989695e-01	[ 1.1559005e+00]	 1.9528913e-01


.. parsed-literal::

      56	 1.1544584e+00	 1.5929815e-01	 1.2031019e+00	 1.5968576e-01	[ 1.1667748e+00]	 2.1851754e-01


.. parsed-literal::

      57	 1.1656684e+00	 1.5876556e-01	 1.2147890e+00	 1.6015784e-01	  1.1664676e+00 	 2.1809459e-01


.. parsed-literal::

      58	 1.1736877e+00	 1.5895096e-01	 1.2229329e+00	 1.6054814e-01	[ 1.1705638e+00]	 2.0548511e-01


.. parsed-literal::

      59	 1.1806143e+00	 1.5805360e-01	 1.2298951e+00	 1.5984821e-01	[ 1.1750929e+00]	 2.1883512e-01


.. parsed-literal::

      60	 1.1914084e+00	 1.5640392e-01	 1.2414087e+00	 1.5827390e-01	[ 1.1772991e+00]	 2.1815872e-01


.. parsed-literal::

      61	 1.1980691e+00	 1.5606082e-01	 1.2482925e+00	 1.5773460e-01	[ 1.1808081e+00]	 2.0439315e-01
      62	 1.2065827e+00	 1.5557618e-01	 1.2570469e+00	 1.5666735e-01	[ 1.1854870e+00]	 1.8386197e-01


.. parsed-literal::

      63	 1.2149855e+00	 1.5484946e-01	 1.2655198e+00	 1.5483545e-01	[ 1.1914447e+00]	 2.1131945e-01


.. parsed-literal::

      64	 1.2218179e+00	 1.5453226e-01	 1.2726623e+00	 1.5370378e-01	[ 1.1919896e+00]	 2.1786118e-01
      65	 1.2276970e+00	 1.5405705e-01	 1.2784246e+00	 1.5320274e-01	[ 1.1970438e+00]	 1.8594885e-01


.. parsed-literal::

      66	 1.2368238e+00	 1.5341882e-01	 1.2878845e+00	 1.5242021e-01	  1.1938085e+00 	 1.8480110e-01


.. parsed-literal::

      67	 1.2434689e+00	 1.5328690e-01	 1.2946395e+00	 1.5211023e-01	  1.1897486e+00 	 2.1870494e-01


.. parsed-literal::

      68	 1.2501518e+00	 1.5235692e-01	 1.3015287e+00	 1.5087307e-01	  1.1892404e+00 	 2.0799804e-01
      69	 1.2550965e+00	 1.5223699e-01	 1.3064136e+00	 1.5059368e-01	  1.1871200e+00 	 2.0254683e-01


.. parsed-literal::

      70	 1.2598282e+00	 1.5177539e-01	 1.3112785e+00	 1.4998154e-01	  1.1909334e+00 	 2.1687675e-01


.. parsed-literal::

      71	 1.2658374e+00	 1.5047036e-01	 1.3175473e+00	 1.4857357e-01	[ 1.1991488e+00]	 2.1072578e-01


.. parsed-literal::

      72	 1.2722254e+00	 1.4962792e-01	 1.3240549e+00	 1.4756678e-01	[ 1.2026310e+00]	 2.1063781e-01
      73	 1.2772884e+00	 1.4874475e-01	 1.3291835e+00	 1.4657947e-01	[ 1.2094524e+00]	 1.9027209e-01


.. parsed-literal::

      74	 1.2853156e+00	 1.4767773e-01	 1.3372775e+00	 1.4536226e-01	[ 1.2256133e+00]	 1.9742227e-01


.. parsed-literal::

      75	 1.2896526e+00	 1.4674530e-01	 1.3418233e+00	 1.4420607e-01	[ 1.2290549e+00]	 3.0087399e-01
      76	 1.2961463e+00	 1.4612758e-01	 1.3482723e+00	 1.4357689e-01	[ 1.2437143e+00]	 1.8151069e-01


.. parsed-literal::

      77	 1.3013819e+00	 1.4501146e-01	 1.3536593e+00	 1.4223646e-01	[ 1.2508541e+00]	 2.0983481e-01


.. parsed-literal::

      78	 1.3067714e+00	 1.4420646e-01	 1.3591571e+00	 1.4147816e-01	[ 1.2552682e+00]	 2.1899223e-01


.. parsed-literal::

      79	 1.3113713e+00	 1.4337205e-01	 1.3637474e+00	 1.4070703e-01	[ 1.2579566e+00]	 2.1752906e-01


.. parsed-literal::

      80	 1.3181003e+00	 1.4252827e-01	 1.3705430e+00	 1.3987010e-01	[ 1.2639436e+00]	 2.1211457e-01


.. parsed-literal::

      81	 1.3225765e+00	 1.4193974e-01	 1.3750788e+00	 1.3934548e-01	[ 1.2656326e+00]	 2.1494436e-01


.. parsed-literal::

      82	 1.3275745e+00	 1.4203102e-01	 1.3800008e+00	 1.3954414e-01	[ 1.2723498e+00]	 2.0358777e-01
      83	 1.3334852e+00	 1.4196339e-01	 1.3860521e+00	 1.3942540e-01	[ 1.2792451e+00]	 1.8560886e-01


.. parsed-literal::

      84	 1.3371759e+00	 1.4197439e-01	 1.3898254e+00	 1.3962937e-01	[ 1.2818842e+00]	 1.8772197e-01


.. parsed-literal::

      85	 1.3415236e+00	 1.4165906e-01	 1.3942491e+00	 1.3928565e-01	  1.2807892e+00 	 2.1199870e-01


.. parsed-literal::

      86	 1.3465432e+00	 1.4142097e-01	 1.3993285e+00	 1.3901885e-01	  1.2808179e+00 	 2.0387697e-01


.. parsed-literal::

      87	 1.3510248e+00	 1.4101985e-01	 1.4038440e+00	 1.3906802e-01	  1.2739582e+00 	 2.1060252e-01
      88	 1.3555186e+00	 1.4077836e-01	 1.4083743e+00	 1.3921698e-01	  1.2728333e+00 	 1.8655586e-01


.. parsed-literal::

      89	 1.3599352e+00	 1.4103285e-01	 1.4129184e+00	 1.4018391e-01	  1.2690593e+00 	 2.0623732e-01
      90	 1.3636021e+00	 1.4075178e-01	 1.4165816e+00	 1.3995579e-01	  1.2740271e+00 	 1.7445803e-01


.. parsed-literal::

      91	 1.3658470e+00	 1.4059017e-01	 1.4188180e+00	 1.3972316e-01	  1.2775912e+00 	 1.8377018e-01


.. parsed-literal::

      92	 1.3721548e+00	 1.3983399e-01	 1.4252352e+00	 1.3918781e-01	  1.2795743e+00 	 2.0857048e-01


.. parsed-literal::

      93	 1.3750755e+00	 1.3941009e-01	 1.4281599e+00	 1.3914099e-01	[ 1.2858855e+00]	 3.1449103e-01


.. parsed-literal::

      94	 1.3791365e+00	 1.3904477e-01	 1.4321955e+00	 1.3920424e-01	  1.2836818e+00 	 2.0662260e-01


.. parsed-literal::

      95	 1.3842239e+00	 1.3826697e-01	 1.4373584e+00	 1.3950145e-01	  1.2748183e+00 	 2.1373987e-01


.. parsed-literal::

      96	 1.3868243e+00	 1.3822486e-01	 1.4399828e+00	 1.3962606e-01	  1.2712311e+00 	 2.0903563e-01


.. parsed-literal::

      97	 1.3896188e+00	 1.3805142e-01	 1.4427704e+00	 1.3952589e-01	  1.2742715e+00 	 2.0946264e-01


.. parsed-literal::

      98	 1.3953787e+00	 1.3740033e-01	 1.4487197e+00	 1.3918649e-01	  1.2736275e+00 	 2.1699119e-01


.. parsed-literal::

      99	 1.3980918e+00	 1.3724164e-01	 1.4514327e+00	 1.3907329e-01	  1.2745043e+00 	 2.1954894e-01
     100	 1.4022434e+00	 1.3680360e-01	 1.4556246e+00	 1.3878534e-01	  1.2729736e+00 	 1.9752097e-01


.. parsed-literal::

     101	 1.4058579e+00	 1.3649172e-01	 1.4593739e+00	 1.3849027e-01	  1.2705198e+00 	 2.0966506e-01
     102	 1.4092708e+00	 1.3627999e-01	 1.4628505e+00	 1.3831684e-01	  1.2710223e+00 	 1.7810774e-01


.. parsed-literal::

     103	 1.4120083e+00	 1.3618298e-01	 1.4656393e+00	 1.3795955e-01	  1.2739190e+00 	 1.6628909e-01


.. parsed-literal::

     104	 1.4151822e+00	 1.3598200e-01	 1.4689846e+00	 1.3767098e-01	  1.2690239e+00 	 2.0555663e-01
     105	 1.4179262e+00	 1.3563219e-01	 1.4718419e+00	 1.3720218e-01	  1.2625463e+00 	 1.9775081e-01


.. parsed-literal::

     106	 1.4205847e+00	 1.3545289e-01	 1.4745037e+00	 1.3699755e-01	  1.2619079e+00 	 1.8946528e-01


.. parsed-literal::

     107	 1.4232907e+00	 1.3520334e-01	 1.4772186e+00	 1.3666713e-01	  1.2605123e+00 	 2.1126294e-01


.. parsed-literal::

     108	 1.4257849e+00	 1.3510769e-01	 1.4798139e+00	 1.3650858e-01	  1.2571719e+00 	 2.1126890e-01


.. parsed-literal::

     109	 1.4280749e+00	 1.3500948e-01	 1.4821562e+00	 1.3602337e-01	  1.2601485e+00 	 2.2034717e-01


.. parsed-literal::

     110	 1.4297133e+00	 1.3496943e-01	 1.4838226e+00	 1.3593057e-01	  1.2579392e+00 	 2.1539497e-01
     111	 1.4323652e+00	 1.3492559e-01	 1.4865275e+00	 1.3582574e-01	  1.2547972e+00 	 1.8889189e-01


.. parsed-literal::

     112	 1.4343492e+00	 1.3481723e-01	 1.4885573e+00	 1.3557409e-01	  1.2549422e+00 	 1.9848633e-01


.. parsed-literal::

     113	 1.4365321e+00	 1.3469889e-01	 1.4907054e+00	 1.3554696e-01	  1.2530538e+00 	 2.1089721e-01
     114	 1.4383900e+00	 1.3457095e-01	 1.4925289e+00	 1.3545100e-01	  1.2538620e+00 	 1.7544746e-01


.. parsed-literal::

     115	 1.4401907e+00	 1.3448527e-01	 1.4942977e+00	 1.3537972e-01	  1.2537856e+00 	 2.0692492e-01
     116	 1.4415437e+00	 1.3418971e-01	 1.4957267e+00	 1.3519149e-01	  1.2585200e+00 	 1.7860842e-01


.. parsed-literal::

     117	 1.4439845e+00	 1.3422704e-01	 1.4981314e+00	 1.3517473e-01	  1.2550451e+00 	 2.0088315e-01


.. parsed-literal::

     118	 1.4453093e+00	 1.3420392e-01	 1.4994789e+00	 1.3513589e-01	  1.2533611e+00 	 2.1196699e-01
     119	 1.4468683e+00	 1.3408246e-01	 1.5011018e+00	 1.3505789e-01	  1.2511787e+00 	 1.8232346e-01


.. parsed-literal::

     120	 1.4493693e+00	 1.3374191e-01	 1.5037123e+00	 1.3494469e-01	  1.2447184e+00 	 2.0194411e-01


.. parsed-literal::

     121	 1.4508330e+00	 1.3348039e-01	 1.5052770e+00	 1.3474524e-01	  1.2474606e+00 	 3.1596041e-01


.. parsed-literal::

     122	 1.4527446e+00	 1.3318319e-01	 1.5072192e+00	 1.3470272e-01	  1.2435232e+00 	 2.0658350e-01
     123	 1.4543696e+00	 1.3293347e-01	 1.5088282e+00	 1.3457618e-01	  1.2432619e+00 	 1.7472863e-01


.. parsed-literal::

     124	 1.4562127e+00	 1.3273845e-01	 1.5106505e+00	 1.3453964e-01	  1.2455646e+00 	 1.8281031e-01


.. parsed-literal::

     125	 1.4583676e+00	 1.3249198e-01	 1.5127830e+00	 1.3419873e-01	  1.2475400e+00 	 2.0479536e-01


.. parsed-literal::

     126	 1.4604381e+00	 1.3231239e-01	 1.5148682e+00	 1.3394419e-01	  1.2488899e+00 	 2.1731758e-01
     127	 1.4628407e+00	 1.3206041e-01	 1.5173645e+00	 1.3350550e-01	  1.2483257e+00 	 1.7790771e-01


.. parsed-literal::

     128	 1.4645264e+00	 1.3174609e-01	 1.5191742e+00	 1.3320681e-01	  1.2478726e+00 	 1.8434858e-01


.. parsed-literal::

     129	 1.4659227e+00	 1.3167164e-01	 1.5205992e+00	 1.3316680e-01	  1.2470814e+00 	 2.0848465e-01


.. parsed-literal::

     130	 1.4682319e+00	 1.3139240e-01	 1.5230647e+00	 1.3308057e-01	  1.2379453e+00 	 2.0925689e-01


.. parsed-literal::

     131	 1.4698022e+00	 1.3130517e-01	 1.5246935e+00	 1.3302957e-01	  1.2356344e+00 	 2.0224810e-01


.. parsed-literal::

     132	 1.4719200e+00	 1.3122396e-01	 1.5268600e+00	 1.3299939e-01	  1.2264328e+00 	 2.1049285e-01


.. parsed-literal::

     133	 1.4743233e+00	 1.3107510e-01	 1.5292943e+00	 1.3280897e-01	  1.2196371e+00 	 2.0646405e-01
     134	 1.4761851e+00	 1.3102316e-01	 1.5310806e+00	 1.3276082e-01	  1.2085914e+00 	 1.8452621e-01


.. parsed-literal::

     135	 1.4776193e+00	 1.3096623e-01	 1.5324543e+00	 1.3268052e-01	  1.2094059e+00 	 2.1134067e-01
     136	 1.4800838e+00	 1.3073305e-01	 1.5349246e+00	 1.3246146e-01	  1.2013067e+00 	 1.9066811e-01


.. parsed-literal::

     137	 1.4804510e+00	 1.3077212e-01	 1.5353719e+00	 1.3252685e-01	  1.2015593e+00 	 2.1422434e-01
     138	 1.4822124e+00	 1.3076376e-01	 1.5370658e+00	 1.3248308e-01	  1.2015703e+00 	 1.9920540e-01


.. parsed-literal::

     139	 1.4830968e+00	 1.3075546e-01	 1.5379838e+00	 1.3247821e-01	  1.1954244e+00 	 2.1335030e-01


.. parsed-literal::

     140	 1.4843610e+00	 1.3081270e-01	 1.5393066e+00	 1.3248984e-01	  1.1877940e+00 	 2.4242067e-01


.. parsed-literal::

     141	 1.4861303e+00	 1.3080872e-01	 1.5411319e+00	 1.3246176e-01	  1.1764146e+00 	 2.0683813e-01
     142	 1.4878355e+00	 1.3100291e-01	 1.5429621e+00	 1.3248467e-01	  1.1646287e+00 	 2.0628309e-01


.. parsed-literal::

     143	 1.4893354e+00	 1.3084755e-01	 1.5444021e+00	 1.3244383e-01	  1.1607404e+00 	 2.0861459e-01


.. parsed-literal::

     144	 1.4902694e+00	 1.3070592e-01	 1.5452884e+00	 1.3238578e-01	  1.1642598e+00 	 2.1122503e-01


.. parsed-literal::

     145	 1.4916049e+00	 1.3053207e-01	 1.5466102e+00	 1.3236937e-01	  1.1645043e+00 	 2.0586967e-01


.. parsed-literal::

     146	 1.4930474e+00	 1.3030977e-01	 1.5481164e+00	 1.3236954e-01	  1.1679245e+00 	 2.1107578e-01


.. parsed-literal::

     147	 1.4949620e+00	 1.3013481e-01	 1.5500282e+00	 1.3230429e-01	  1.1631690e+00 	 2.1116567e-01


.. parsed-literal::

     148	 1.4963218e+00	 1.3003695e-01	 1.5514295e+00	 1.3225917e-01	  1.1571936e+00 	 2.0237470e-01


.. parsed-literal::

     149	 1.4975700e+00	 1.2991498e-01	 1.5527428e+00	 1.3216452e-01	  1.1535620e+00 	 2.1201897e-01


.. parsed-literal::

     150	 1.4984596e+00	 1.2954975e-01	 1.5538335e+00	 1.3187710e-01	  1.1310327e+00 	 2.0892715e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff13894d1e0>



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
    CPU times: user 1.75 s, sys: 46 ms, total: 1.8 s
    Wall time: 556 ms


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

