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
       1	-3.4535546e-01	 3.2102671e-01	-3.3561169e-01	 3.1833109e-01	[-3.3097002e-01]	 4.7007585e-01


.. parsed-literal::

       2	-2.7126438e-01	 3.0901424e-01	-2.4567363e-01	 3.0752706e-01	[-2.4175681e-01]	 2.3169875e-01


.. parsed-literal::

       3	-2.2732134e-01	 2.8890021e-01	-1.8422038e-01	 2.9011389e-01	[-1.9103522e-01]	 2.8615379e-01
       4	-1.9906916e-01	 2.6472475e-01	-1.5826658e-01	 2.7014633e-01	[-1.9084382e-01]	 1.9035697e-01


.. parsed-literal::

       5	-1.0438720e-01	 2.5666843e-01	-6.6768910e-02	 2.5976434e-01	[-8.2523669e-02]	 2.0618248e-01
       6	-7.1599471e-02	 2.5160953e-01	-3.9197982e-02	 2.5522309e-01	[-5.2583453e-02]	 1.8540049e-01


.. parsed-literal::

       7	-5.1100121e-02	 2.4831881e-01	-2.6177451e-02	 2.5136341e-01	[-3.8675526e-02]	 1.9605780e-01
       8	-4.0557368e-02	 2.4666445e-01	-1.9267013e-02	 2.4921687e-01	[-3.0225479e-02]	 2.0707679e-01


.. parsed-literal::

       9	-2.7006578e-02	 2.4422973e-01	-8.8995475e-03	 2.4647180e-01	[-1.8788852e-02]	 1.9350004e-01
      10	-1.4586245e-02	 2.4171379e-01	 1.2625276e-03	 2.4450958e-01	[-1.1500580e-02]	 1.9419360e-01


.. parsed-literal::

      11	-1.2569459e-02	 2.4108525e-01	 1.6719114e-03	 2.4514214e-01	 -1.3309713e-02 	 2.0991158e-01


.. parsed-literal::

      12	-5.7103628e-03	 2.4036362e-01	 8.3631336e-03	 2.4424342e-01	[-7.4312667e-03]	 2.1889734e-01


.. parsed-literal::

      13	-3.1837841e-03	 2.3981464e-01	 1.0871791e-02	 2.4353676e-01	[-4.5860579e-03]	 2.1313715e-01


.. parsed-literal::

      14	 4.8219972e-04	 2.3898003e-01	 1.4857830e-02	 2.4264195e-01	[-1.2347147e-04]	 2.0111680e-01


.. parsed-literal::

      15	 1.4579053e-01	 2.2102655e-01	 1.6839334e-01	 2.2396649e-01	[ 1.6363073e-01]	 4.3993044e-01


.. parsed-literal::

      16	 2.5075854e-01	 2.1471628e-01	 2.7570052e-01	 2.1837041e-01	[ 2.6278483e-01]	 2.2101355e-01
      17	 2.7867560e-01	 2.0979270e-01	 3.0942928e-01	 2.1408710e-01	[ 2.6998390e-01]	 1.9048333e-01


.. parsed-literal::

      18	 3.3834743e-01	 2.1362829e-01	 3.6831697e-01	 2.1722378e-01	[ 3.4809215e-01]	 2.1929741e-01


.. parsed-literal::

      19	 3.7061474e-01	 2.0891622e-01	 4.0140699e-01	 2.1398866e-01	[ 3.7965446e-01]	 2.1758056e-01


.. parsed-literal::

      20	 4.3312453e-01	 2.0606498e-01	 4.6471264e-01	 2.1137675e-01	[ 4.4203725e-01]	 2.0607519e-01


.. parsed-literal::

      21	 5.2433837e-01	 2.0889325e-01	 5.5869150e-01	 2.1243506e-01	[ 5.3327720e-01]	 2.1002102e-01


.. parsed-literal::

      22	 5.8438274e-01	 2.0821562e-01	 6.2349087e-01	 2.1100908e-01	[ 5.7730083e-01]	 2.1127295e-01


.. parsed-literal::

      23	 6.2584821e-01	 2.0347884e-01	 6.6391054e-01	 2.0678212e-01	[ 6.2067107e-01]	 2.1483350e-01
      24	 6.6358959e-01	 1.9992924e-01	 7.0053480e-01	 2.0349547e-01	[ 6.6192169e-01]	 1.7683792e-01


.. parsed-literal::

      25	 7.0932065e-01	 1.9377695e-01	 7.4451177e-01	 1.9484616e-01	[ 7.1683043e-01]	 1.9521761e-01
      26	 7.3273281e-01	 1.9361574e-01	 7.6824040e-01	 1.9479977e-01	[ 7.4204219e-01]	 1.9810295e-01


.. parsed-literal::

      27	 7.5253214e-01	 1.9407165e-01	 7.8973496e-01	 1.9418569e-01	[ 7.4732615e-01]	 1.7356539e-01
      28	 7.7500056e-01	 1.9419307e-01	 8.1212224e-01	 1.9388559e-01	[ 7.7717225e-01]	 1.9008040e-01


.. parsed-literal::

      29	 8.0109186e-01	 1.9451815e-01	 8.3917394e-01	 1.9505445e-01	[ 8.1413582e-01]	 2.0810437e-01


.. parsed-literal::

      30	 8.3148082e-01	 1.9226147e-01	 8.7045992e-01	 1.9292951e-01	[ 8.4902345e-01]	 2.0953608e-01
      31	 8.5336949e-01	 1.9010500e-01	 8.9288120e-01	 1.9070001e-01	[ 8.7197836e-01]	 1.7142534e-01


.. parsed-literal::

      32	 8.7356993e-01	 1.9030310e-01	 9.1327189e-01	 1.9102419e-01	[ 8.9564877e-01]	 1.9458771e-01


.. parsed-literal::

      33	 8.9266864e-01	 1.9135164e-01	 9.3346414e-01	 1.9173280e-01	[ 9.2949679e-01]	 2.1234345e-01


.. parsed-literal::

      34	 9.1189687e-01	 1.8938142e-01	 9.5288486e-01	 1.8981462e-01	[ 9.3788596e-01]	 2.1402884e-01


.. parsed-literal::

      35	 9.2858349e-01	 1.8714731e-01	 9.6987657e-01	 1.8798333e-01	[ 9.4442582e-01]	 2.1362519e-01


.. parsed-literal::

      36	 9.4294203e-01	 1.8444235e-01	 9.8487786e-01	 1.8674207e-01	[ 9.5298443e-01]	 2.1531296e-01


.. parsed-literal::

      37	 9.5867195e-01	 1.8308615e-01	 1.0009765e+00	 1.8632567e-01	[ 9.6957943e-01]	 2.1833062e-01


.. parsed-literal::

      38	 9.7170802e-01	 1.8177255e-01	 1.0142930e+00	 1.8570668e-01	[ 9.8580856e-01]	 2.2391272e-01
      39	 9.8439957e-01	 1.8084130e-01	 1.0278054e+00	 1.8539649e-01	[ 9.9780158e-01]	 1.9322777e-01


.. parsed-literal::

      40	 1.0005722e+00	 1.7939224e-01	 1.0453656e+00	 1.8406141e-01	[ 1.0134085e+00]	 2.0497012e-01


.. parsed-literal::

      41	 1.0152069e+00	 1.8114100e-01	 1.0600269e+00	 1.8571821e-01	[ 1.0251567e+00]	 2.1173477e-01
      42	 1.0264658e+00	 1.8111684e-01	 1.0714530e+00	 1.8537002e-01	[ 1.0343684e+00]	 1.8432212e-01


.. parsed-literal::

      43	 1.0345673e+00	 1.8165009e-01	 1.0797133e+00	 1.8582806e-01	[ 1.0374173e+00]	 1.9071531e-01


.. parsed-literal::

      44	 1.0425648e+00	 1.8190793e-01	 1.0875007e+00	 1.8539553e-01	[ 1.0478453e+00]	 2.0803666e-01


.. parsed-literal::

      45	 1.0522336e+00	 1.8087014e-01	 1.0972312e+00	 1.8423245e-01	[ 1.0539824e+00]	 2.1320486e-01


.. parsed-literal::

      46	 1.0655886e+00	 1.7875191e-01	 1.1110433e+00	 1.8184268e-01	[ 1.0575596e+00]	 2.0383024e-01


.. parsed-literal::

      47	 1.0723523e+00	 1.7771703e-01	 1.1180468e+00	 1.8031047e-01	[ 1.0643077e+00]	 2.0949459e-01


.. parsed-literal::

      48	 1.0804706e+00	 1.7692657e-01	 1.1260590e+00	 1.7972713e-01	[ 1.0736332e+00]	 2.1947122e-01


.. parsed-literal::

      49	 1.0873096e+00	 1.7560175e-01	 1.1331600e+00	 1.7873524e-01	[ 1.0810273e+00]	 2.0274973e-01


.. parsed-literal::

      50	 1.0950858e+00	 1.7379464e-01	 1.1412688e+00	 1.7731156e-01	[ 1.0884914e+00]	 2.1479321e-01


.. parsed-literal::

      51	 1.1072825e+00	 1.7088475e-01	 1.1539244e+00	 1.7469857e-01	[ 1.1016938e+00]	 2.0926762e-01


.. parsed-literal::

      52	 1.1138556e+00	 1.6908135e-01	 1.1609346e+00	 1.7294999e-01	[ 1.1067643e+00]	 2.9469991e-01


.. parsed-literal::

      53	 1.1210027e+00	 1.6765306e-01	 1.1682382e+00	 1.7147194e-01	[ 1.1151004e+00]	 2.2068095e-01


.. parsed-literal::

      54	 1.1285894e+00	 1.6622088e-01	 1.1760215e+00	 1.7007103e-01	[ 1.1223079e+00]	 2.0583701e-01
      55	 1.1407049e+00	 1.6299769e-01	 1.1886613e+00	 1.6728709e-01	[ 1.1333035e+00]	 1.7903137e-01


.. parsed-literal::

      56	 1.1473470e+00	 1.5988477e-01	 1.1959498e+00	 1.6492219e-01	[ 1.1356869e+00]	 2.1343422e-01
      57	 1.1568880e+00	 1.5938175e-01	 1.2051712e+00	 1.6426378e-01	[ 1.1481960e+00]	 1.7910409e-01


.. parsed-literal::

      58	 1.1633394e+00	 1.5853596e-01	 1.2116722e+00	 1.6360437e-01	[ 1.1555467e+00]	 1.7345476e-01
      59	 1.1704289e+00	 1.5731693e-01	 1.2190316e+00	 1.6248708e-01	[ 1.1628827e+00]	 1.8236184e-01


.. parsed-literal::

      60	 1.1788081e+00	 1.5618167e-01	 1.2276306e+00	 1.6152658e-01	[ 1.1688014e+00]	 2.0086384e-01
      61	 1.1846238e+00	 1.5409004e-01	 1.2336568e+00	 1.5960916e-01	[ 1.1740365e+00]	 2.0511842e-01


.. parsed-literal::

      62	 1.1905385e+00	 1.5373701e-01	 1.2395129e+00	 1.5881265e-01	[ 1.1803748e+00]	 2.0537472e-01


.. parsed-literal::

      63	 1.1988195e+00	 1.5262683e-01	 1.2480754e+00	 1.5747265e-01	[ 1.1869579e+00]	 2.0782828e-01


.. parsed-literal::

      64	 1.2068448e+00	 1.5117627e-01	 1.2565900e+00	 1.5604333e-01	[ 1.1912671e+00]	 2.0896292e-01


.. parsed-literal::

      65	 1.2139116e+00	 1.5078284e-01	 1.2636498e+00	 1.5611350e-01	[ 1.1955844e+00]	 2.0096636e-01
      66	 1.2194234e+00	 1.5020774e-01	 1.2691599e+00	 1.5625682e-01	[ 1.1974995e+00]	 1.8660760e-01


.. parsed-literal::

      67	 1.2275260e+00	 1.4898572e-01	 1.2776287e+00	 1.5596618e-01	[ 1.1995196e+00]	 1.9068813e-01


.. parsed-literal::

      68	 1.2336750e+00	 1.4756480e-01	 1.2841884e+00	 1.5584185e-01	[ 1.1998550e+00]	 2.1081257e-01


.. parsed-literal::

      69	 1.2407836e+00	 1.4711026e-01	 1.2913291e+00	 1.5532601e-01	[ 1.2065455e+00]	 2.1907783e-01
      70	 1.2455101e+00	 1.4639573e-01	 1.2961713e+00	 1.5451110e-01	[ 1.2125122e+00]	 1.9721627e-01


.. parsed-literal::

      71	 1.2526995e+00	 1.4589032e-01	 1.3035959e+00	 1.5378906e-01	[ 1.2204784e+00]	 2.1741414e-01
      72	 1.2609472e+00	 1.4496719e-01	 1.3123653e+00	 1.5316949e-01	[ 1.2301645e+00]	 1.9770479e-01


.. parsed-literal::

      73	 1.2690172e+00	 1.4445953e-01	 1.3204362e+00	 1.5306756e-01	[ 1.2400679e+00]	 1.7733932e-01


.. parsed-literal::

      74	 1.2737638e+00	 1.4440717e-01	 1.3252680e+00	 1.5354989e-01	[ 1.2419064e+00]	 2.0946908e-01


.. parsed-literal::

      75	 1.2781527e+00	 1.4435446e-01	 1.3297588e+00	 1.5413135e-01	[ 1.2465314e+00]	 2.0671773e-01
      76	 1.2830508e+00	 1.4453437e-01	 1.3347550e+00	 1.5456363e-01	[ 1.2508821e+00]	 1.9238210e-01


.. parsed-literal::

      77	 1.2888272e+00	 1.4474746e-01	 1.3406525e+00	 1.5506878e-01	[ 1.2560923e+00]	 2.0934772e-01


.. parsed-literal::

      78	 1.2949215e+00	 1.4482028e-01	 1.3467311e+00	 1.5520197e-01	[ 1.2632620e+00]	 2.2301078e-01


.. parsed-literal::

      79	 1.2996661e+00	 1.4549639e-01	 1.3516574e+00	 1.5558378e-01	[ 1.2665686e+00]	 2.0889235e-01


.. parsed-literal::

      80	 1.3042288e+00	 1.4487007e-01	 1.3561364e+00	 1.5495040e-01	[ 1.2717432e+00]	 2.1360946e-01


.. parsed-literal::

      81	 1.3099159e+00	 1.4425961e-01	 1.3618630e+00	 1.5431632e-01	[ 1.2756180e+00]	 2.0378041e-01
      82	 1.3153054e+00	 1.4394760e-01	 1.3674287e+00	 1.5418969e-01	[ 1.2807663e+00]	 1.9978404e-01


.. parsed-literal::

      83	 1.3196784e+00	 1.4402673e-01	 1.3722960e+00	 1.5423054e-01	[ 1.2808641e+00]	 2.0175099e-01


.. parsed-literal::

      84	 1.3260750e+00	 1.4359275e-01	 1.3786018e+00	 1.5416317e-01	[ 1.2885413e+00]	 2.1663928e-01
      85	 1.3292544e+00	 1.4337845e-01	 1.3817150e+00	 1.5395796e-01	[ 1.2916625e+00]	 1.7699862e-01


.. parsed-literal::

      86	 1.3347287e+00	 1.4280308e-01	 1.3873259e+00	 1.5334322e-01	[ 1.2937727e+00]	 2.0387745e-01
      87	 1.3376895e+00	 1.4232344e-01	 1.3907126e+00	 1.5289278e-01	  1.2935096e+00 	 1.8736339e-01


.. parsed-literal::

      88	 1.3427698e+00	 1.4192208e-01	 1.3956502e+00	 1.5241141e-01	[ 1.2943483e+00]	 2.1179032e-01


.. parsed-literal::

      89	 1.3462641e+00	 1.4153363e-01	 1.3991938e+00	 1.5199041e-01	  1.2934282e+00 	 2.1199799e-01
      90	 1.3490576e+00	 1.4127411e-01	 1.4020305e+00	 1.5162236e-01	[ 1.2943659e+00]	 1.8130136e-01


.. parsed-literal::

      91	 1.3545755e+00	 1.4096284e-01	 1.4077112e+00	 1.5115152e-01	[ 1.2968972e+00]	 2.0319128e-01


.. parsed-literal::

      92	 1.3581085e+00	 1.4082042e-01	 1.4113267e+00	 1.5061336e-01	[ 1.3032005e+00]	 2.9167438e-01


.. parsed-literal::

      93	 1.3619172e+00	 1.4073062e-01	 1.4152037e+00	 1.5052652e-01	[ 1.3047889e+00]	 2.1333289e-01


.. parsed-literal::

      94	 1.3660041e+00	 1.4054761e-01	 1.4193762e+00	 1.5047780e-01	[ 1.3056244e+00]	 2.0091701e-01
      95	 1.3697940e+00	 1.4040012e-01	 1.4232676e+00	 1.5039107e-01	  1.3055275e+00 	 1.8272328e-01


.. parsed-literal::

      96	 1.3736022e+00	 1.3989982e-01	 1.4272885e+00	 1.5006405e-01	  1.2995668e+00 	 2.0680046e-01


.. parsed-literal::

      97	 1.3771738e+00	 1.3955706e-01	 1.4308372e+00	 1.4966440e-01	  1.3009778e+00 	 2.0968556e-01


.. parsed-literal::

      98	 1.3816751e+00	 1.3906278e-01	 1.4353497e+00	 1.4884946e-01	  1.3023644e+00 	 2.1119714e-01
      99	 1.3850863e+00	 1.3874460e-01	 1.4388690e+00	 1.4837285e-01	  1.3028971e+00 	 1.9926667e-01


.. parsed-literal::

     100	 1.3896336e+00	 1.3842178e-01	 1.4434413e+00	 1.4788593e-01	  1.3042096e+00 	 2.1792936e-01
     101	 1.3937253e+00	 1.3811080e-01	 1.4475703e+00	 1.4740907e-01	[ 1.3096425e+00]	 1.8899298e-01


.. parsed-literal::

     102	 1.3970874e+00	 1.3809126e-01	 1.4509339e+00	 1.4735107e-01	  1.3092645e+00 	 2.1586537e-01


.. parsed-literal::

     103	 1.3993065e+00	 1.3810171e-01	 1.4531050e+00	 1.4739952e-01	[ 1.3126781e+00]	 2.1213245e-01
     104	 1.4041910e+00	 1.3804198e-01	 1.4581486e+00	 1.4718607e-01	[ 1.3154607e+00]	 2.0036459e-01


.. parsed-literal::

     105	 1.4076120e+00	 1.3800694e-01	 1.4616541e+00	 1.4654975e-01	[ 1.3228151e+00]	 2.1594214e-01
     106	 1.4112879e+00	 1.3787562e-01	 1.4653055e+00	 1.4639545e-01	[ 1.3241572e+00]	 1.7738104e-01


.. parsed-literal::

     107	 1.4144293e+00	 1.3770333e-01	 1.4683774e+00	 1.4614075e-01	[ 1.3258169e+00]	 2.0145607e-01


.. parsed-literal::

     108	 1.4183884e+00	 1.3743909e-01	 1.4722863e+00	 1.4586899e-01	[ 1.3287005e+00]	 2.1786737e-01
     109	 1.4196880e+00	 1.3735860e-01	 1.4736707e+00	 1.4580795e-01	[ 1.3335652e+00]	 1.8018794e-01


.. parsed-literal::

     110	 1.4240904e+00	 1.3718469e-01	 1.4779908e+00	 1.4578377e-01	[ 1.3362015e+00]	 1.7877078e-01


.. parsed-literal::

     111	 1.4256567e+00	 1.3711739e-01	 1.4796062e+00	 1.4577463e-01	[ 1.3368641e+00]	 2.1705914e-01


.. parsed-literal::

     112	 1.4282282e+00	 1.3698056e-01	 1.4822771e+00	 1.4574249e-01	[ 1.3387920e+00]	 2.0507908e-01
     113	 1.4318424e+00	 1.3670329e-01	 1.4860390e+00	 1.4554727e-01	[ 1.3395776e+00]	 1.8305469e-01


.. parsed-literal::

     114	 1.4334661e+00	 1.3645708e-01	 1.4878947e+00	 1.4539115e-01	[ 1.3406202e+00]	 2.1351933e-01


.. parsed-literal::

     115	 1.4366538e+00	 1.3629503e-01	 1.4909075e+00	 1.4515947e-01	[ 1.3421582e+00]	 2.1043801e-01


.. parsed-literal::

     116	 1.4377721e+00	 1.3618328e-01	 1.4919641e+00	 1.4501857e-01	[ 1.3423070e+00]	 2.1190190e-01
     117	 1.4398813e+00	 1.3589117e-01	 1.4940607e+00	 1.4469873e-01	  1.3419774e+00 	 1.8232870e-01


.. parsed-literal::

     118	 1.4423388e+00	 1.3532685e-01	 1.4966142e+00	 1.4432960e-01	  1.3374611e+00 	 2.1086955e-01
     119	 1.4452463e+00	 1.3493360e-01	 1.4996402e+00	 1.4388025e-01	  1.3368927e+00 	 1.9382286e-01


.. parsed-literal::

     120	 1.4471880e+00	 1.3474257e-01	 1.5017078e+00	 1.4367925e-01	  1.3367885e+00 	 2.1663880e-01


.. parsed-literal::

     121	 1.4499147e+00	 1.3444197e-01	 1.5046545e+00	 1.4330957e-01	  1.3349444e+00 	 2.0391297e-01


.. parsed-literal::

     122	 1.4522886e+00	 1.3404631e-01	 1.5073063e+00	 1.4277775e-01	  1.3303277e+00 	 2.1205163e-01


.. parsed-literal::

     123	 1.4549949e+00	 1.3377303e-01	 1.5100494e+00	 1.4246415e-01	  1.3285849e+00 	 2.0770574e-01


.. parsed-literal::

     124	 1.4570332e+00	 1.3352878e-01	 1.5120268e+00	 1.4224658e-01	  1.3278798e+00 	 2.0965075e-01


.. parsed-literal::

     125	 1.4595030e+00	 1.3317900e-01	 1.5144512e+00	 1.4200641e-01	  1.3278942e+00 	 2.1059608e-01


.. parsed-literal::

     126	 1.4623022e+00	 1.3262368e-01	 1.5173468e+00	 1.4159321e-01	  1.3260382e+00 	 2.1111655e-01


.. parsed-literal::

     127	 1.4653462e+00	 1.3225611e-01	 1.5204300e+00	 1.4139478e-01	  1.3289107e+00 	 2.1473074e-01
     128	 1.4671030e+00	 1.3215272e-01	 1.5222299e+00	 1.4133237e-01	  1.3303347e+00 	 1.8965936e-01


.. parsed-literal::

     129	 1.4688641e+00	 1.3221269e-01	 1.5240624e+00	 1.4138306e-01	  1.3298644e+00 	 1.9903374e-01


.. parsed-literal::

     130	 1.4703963e+00	 1.3216315e-01	 1.5256328e+00	 1.4132013e-01	  1.3281741e+00 	 2.0808649e-01


.. parsed-literal::

     131	 1.4721596e+00	 1.3217051e-01	 1.5273938e+00	 1.4134082e-01	  1.3256112e+00 	 2.1020293e-01
     132	 1.4739401e+00	 1.3213961e-01	 1.5291401e+00	 1.4135978e-01	  1.3238982e+00 	 1.8881106e-01


.. parsed-literal::

     133	 1.4751409e+00	 1.3204076e-01	 1.5302889e+00	 1.4156002e-01	  1.3183932e+00 	 2.0927429e-01


.. parsed-literal::

     134	 1.4776702e+00	 1.3195154e-01	 1.5327236e+00	 1.4142240e-01	  1.3224516e+00 	 2.1641850e-01


.. parsed-literal::

     135	 1.4790164e+00	 1.3183364e-01	 1.5340287e+00	 1.4133329e-01	  1.3247613e+00 	 2.1010423e-01


.. parsed-literal::

     136	 1.4804215e+00	 1.3173990e-01	 1.5354352e+00	 1.4126684e-01	  1.3255153e+00 	 2.2314477e-01
     137	 1.4830505e+00	 1.3164144e-01	 1.5381581e+00	 1.4110983e-01	  1.3243771e+00 	 2.0015383e-01


.. parsed-literal::

     138	 1.4857425e+00	 1.3131513e-01	 1.5410987e+00	 1.4077751e-01	  1.3163419e+00 	 2.0674992e-01


.. parsed-literal::

     139	 1.4878242e+00	 1.3122634e-01	 1.5433552e+00	 1.4057183e-01	  1.3110721e+00 	 2.0796084e-01


.. parsed-literal::

     140	 1.4893585e+00	 1.3111927e-01	 1.5447927e+00	 1.4047718e-01	  1.3134891e+00 	 2.1173644e-01


.. parsed-literal::

     141	 1.4906279e+00	 1.3092858e-01	 1.5460639e+00	 1.4032449e-01	  1.3118662e+00 	 2.1089935e-01


.. parsed-literal::

     142	 1.4919325e+00	 1.3068053e-01	 1.5474466e+00	 1.4010387e-01	  1.3099983e+00 	 2.1099234e-01


.. parsed-literal::

     143	 1.4934868e+00	 1.3043620e-01	 1.5490317e+00	 1.3989775e-01	  1.3072287e+00 	 2.0804763e-01
     144	 1.4948767e+00	 1.3026522e-01	 1.5504391e+00	 1.3975630e-01	  1.3056967e+00 	 1.8875718e-01


.. parsed-literal::

     145	 1.4960738e+00	 1.3016886e-01	 1.5516789e+00	 1.3969716e-01	  1.3033874e+00 	 1.9290447e-01


.. parsed-literal::

     146	 1.4977723e+00	 1.3005769e-01	 1.5534604e+00	 1.3958845e-01	  1.3027540e+00 	 2.1181989e-01
     147	 1.4994605e+00	 1.2992942e-01	 1.5552390e+00	 1.3952179e-01	  1.2970380e+00 	 1.9077969e-01


.. parsed-literal::

     148	 1.5012689e+00	 1.2985830e-01	 1.5571387e+00	 1.3945830e-01	  1.2930854e+00 	 2.1376610e-01


.. parsed-literal::

     149	 1.5031609e+00	 1.2967798e-01	 1.5590850e+00	 1.3936866e-01	  1.2874284e+00 	 2.0215869e-01


.. parsed-literal::

     150	 1.5049066e+00	 1.2956669e-01	 1.5608755e+00	 1.3935820e-01	  1.2808684e+00 	 2.0344830e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.18 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f917094d3c0>



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
    CPU times: user 2.04 s, sys: 48 ms, total: 2.09 s
    Wall time: 629 ms


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

