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
       1	-3.5017981e-01	 3.2298463e-01	-3.4051183e-01	 3.1186146e-01	[-3.1986747e-01]	 4.6119666e-01


.. parsed-literal::

       2	-2.8016064e-01	 3.1236475e-01	-2.5607158e-01	 3.0043006e-01	[-2.2085678e-01]	 2.3195910e-01


.. parsed-literal::

       3	-2.3281986e-01	 2.8949044e-01	-1.8814938e-01	 2.8171387e-01	[-1.5223718e-01]	 2.8967571e-01
       4	-2.0687396e-01	 2.6628336e-01	-1.6611004e-01	 2.6494473e-01	[-1.4939738e-01]	 1.8592072e-01


.. parsed-literal::

       5	-1.0744391e-01	 2.5847907e-01	-7.2065468e-02	 2.5431479e-01	[-5.2043444e-02]	 2.0061064e-01


.. parsed-literal::

       6	-7.6447875e-02	 2.5312942e-01	-4.5662790e-02	 2.5055670e-01	[-3.1481549e-02]	 2.0122838e-01
       7	-5.5989695e-02	 2.4984684e-01	-3.2451344e-02	 2.4628736e-01	[-1.7183928e-02]	 2.0952296e-01


.. parsed-literal::

       8	-4.4436899e-02	 2.4790856e-01	-2.4320447e-02	 2.4434259e-01	[-9.3986594e-03]	 2.0764756e-01


.. parsed-literal::

       9	-3.0175172e-02	 2.4518978e-01	-1.2790821e-02	 2.4261694e-01	[-1.7508061e-03]	 2.0443106e-01


.. parsed-literal::

      10	-1.9224802e-02	 2.4292098e-01	-3.6755246e-03	 2.4095136e-01	[ 6.7444562e-03]	 2.0415664e-01


.. parsed-literal::

      11	-1.4750235e-02	 2.4233242e-01	-3.5152320e-04	 2.4150576e-01	  5.3852775e-03 	 2.1521711e-01


.. parsed-literal::

      12	-1.0898688e-02	 2.4159315e-01	 3.2523473e-03	 2.4067212e-01	[ 9.6816632e-03]	 2.0921254e-01
      13	-7.2231714e-03	 2.4082055e-01	 6.9386455e-03	 2.3953602e-01	[ 1.5174139e-02]	 1.7575979e-01


.. parsed-literal::

      14	-1.3979963e-03	 2.3947203e-01	 1.3722627e-02	 2.3779565e-01	[ 2.4682261e-02]	 1.7860389e-01


.. parsed-literal::

      15	 1.0195547e-01	 2.2702114e-01	 1.2229537e-01	 2.2021062e-01	[ 1.5261199e-01]	 2.1502233e-01


.. parsed-literal::

      16	 1.6648804e-01	 2.2251622e-01	 1.8978138e-01	 2.1445867e-01	[ 2.1087786e-01]	 2.9199219e-01


.. parsed-literal::

      17	 2.3200409e-01	 2.1869602e-01	 2.5780833e-01	 2.1309222e-01	[ 2.7295699e-01]	 2.0972729e-01


.. parsed-literal::

      18	 3.0129083e-01	 2.1450707e-01	 3.3138431e-01	 2.1007679e-01	[ 3.5094489e-01]	 2.1652889e-01


.. parsed-literal::

      19	 3.3023190e-01	 2.1272411e-01	 3.6254389e-01	 2.0947317e-01	[ 3.7554432e-01]	 2.1707487e-01
      20	 3.7406254e-01	 2.1092974e-01	 4.0732791e-01	 2.0835232e-01	[ 4.2076560e-01]	 1.9802499e-01


.. parsed-literal::

      21	 4.1618918e-01	 2.0910434e-01	 4.4929819e-01	 2.0440068e-01	[ 4.6957775e-01]	 2.0800257e-01
      22	 4.8291364e-01	 2.1058688e-01	 5.1728791e-01	 2.0703800e-01	[ 5.4253797e-01]	 1.9890094e-01


.. parsed-literal::

      23	 5.5554125e-01	 2.0747444e-01	 5.9238299e-01	 2.0254784e-01	[ 6.1809970e-01]	 2.0773792e-01


.. parsed-literal::

      24	 5.9164545e-01	 2.0593390e-01	 6.3054723e-01	 2.0115830e-01	[ 6.5335035e-01]	 2.1107697e-01
      25	 6.2443345e-01	 2.0302159e-01	 6.6229100e-01	 1.9878979e-01	[ 6.7665971e-01]	 1.8300891e-01


.. parsed-literal::

      26	 6.5772812e-01	 2.0258435e-01	 6.9525506e-01	 1.9772980e-01	[ 7.0697641e-01]	 2.1519852e-01


.. parsed-literal::

      27	 7.0115390e-01	 2.0075723e-01	 7.3694633e-01	 1.9446091e-01	[ 7.4803209e-01]	 2.1977544e-01


.. parsed-literal::

      28	 7.3381702e-01	 2.0072912e-01	 7.7027472e-01	 1.9158322e-01	[ 7.8103856e-01]	 2.1983647e-01


.. parsed-literal::

      29	 7.5721338e-01	 2.0025502e-01	 7.9692952e-01	 1.9075563e-01	[ 8.0684625e-01]	 2.1060324e-01
      30	 7.8246651e-01	 2.0065252e-01	 8.2104479e-01	 1.8948436e-01	[ 8.3740346e-01]	 1.8163824e-01


.. parsed-literal::

      31	 8.0136640e-01	 1.9797944e-01	 8.3962162e-01	 1.8800965e-01	[ 8.4826228e-01]	 2.1533775e-01


.. parsed-literal::

      32	 8.2234563e-01	 1.9800760e-01	 8.6096366e-01	 1.8801902e-01	[ 8.7374526e-01]	 2.1039295e-01


.. parsed-literal::

      33	 8.4871263e-01	 1.9812756e-01	 8.8969140e-01	 1.8951054e-01	[ 9.0627055e-01]	 2.1293402e-01


.. parsed-literal::

      34	 8.6948669e-01	 1.9593137e-01	 9.1021177e-01	 1.8785689e-01	[ 9.2725491e-01]	 2.0687246e-01


.. parsed-literal::

      35	 8.8646464e-01	 1.9389009e-01	 9.2752402e-01	 1.8680975e-01	[ 9.4124819e-01]	 2.0898795e-01
      36	 9.0900043e-01	 1.9079970e-01	 9.5059900e-01	 1.8497050e-01	[ 9.6027241e-01]	 1.9362211e-01


.. parsed-literal::

      37	 9.2291456e-01	 1.8740688e-01	 9.6560592e-01	 1.8284331e-01	[ 9.6402887e-01]	 2.0350027e-01


.. parsed-literal::

      38	 9.3875497e-01	 1.8523712e-01	 9.8134940e-01	 1.8013510e-01	[ 9.7884208e-01]	 2.1724868e-01
      39	 9.4886523e-01	 1.8426275e-01	 9.9146548e-01	 1.7908610e-01	[ 9.8922637e-01]	 1.9140983e-01


.. parsed-literal::

      40	 9.6669220e-01	 1.8273929e-01	 1.0101778e+00	 1.7819474e-01	[ 1.0036126e+00]	 2.0762968e-01


.. parsed-literal::

      41	 9.8380200e-01	 1.8196345e-01	 1.0294478e+00	 1.7821253e-01	[ 1.0151466e+00]	 2.0092058e-01


.. parsed-literal::

      42	 9.9921268e-01	 1.8016246e-01	 1.0446799e+00	 1.7674814e-01	[ 1.0306248e+00]	 2.1922064e-01
      43	 1.0064104e+00	 1.7909056e-01	 1.0520041e+00	 1.7609229e-01	[ 1.0347517e+00]	 2.0270395e-01


.. parsed-literal::

      44	 1.0268799e+00	 1.7635975e-01	 1.0734542e+00	 1.7516301e-01	[ 1.0487543e+00]	 1.6512156e-01


.. parsed-literal::

      45	 1.0390043e+00	 1.7527084e-01	 1.0858062e+00	 1.7516813e-01	[ 1.0591454e+00]	 2.1610570e-01


.. parsed-literal::

      46	 1.0501076e+00	 1.7488648e-01	 1.0971847e+00	 1.7552578e-01	[ 1.0754601e+00]	 2.1086693e-01
      47	 1.0580920e+00	 1.7447922e-01	 1.1051152e+00	 1.7513961e-01	[ 1.0863519e+00]	 2.0203471e-01


.. parsed-literal::

      48	 1.0706415e+00	 1.7286095e-01	 1.1177720e+00	 1.7390422e-01	[ 1.0994770e+00]	 2.1062875e-01


.. parsed-literal::

      49	 1.0864032e+00	 1.6874306e-01	 1.1335987e+00	 1.7044928e-01	[ 1.1175488e+00]	 2.1789694e-01


.. parsed-literal::

      50	 1.0946717e+00	 1.6578720e-01	 1.1417801e+00	 1.6811046e-01	[ 1.1286217e+00]	 2.0706129e-01
      51	 1.1025587e+00	 1.6497154e-01	 1.1495153e+00	 1.6714904e-01	[ 1.1355375e+00]	 1.9687915e-01


.. parsed-literal::

      52	 1.1115426e+00	 1.6388010e-01	 1.1587754e+00	 1.6608881e-01	[ 1.1452728e+00]	 1.7412043e-01


.. parsed-literal::

      53	 1.1206360e+00	 1.6229943e-01	 1.1680237e+00	 1.6433615e-01	[ 1.1553447e+00]	 2.0253253e-01
      54	 1.1302931e+00	 1.5821636e-01	 1.1783414e+00	 1.6090867e-01	[ 1.1745295e+00]	 1.7047811e-01


.. parsed-literal::

      55	 1.1437012e+00	 1.5733874e-01	 1.1917773e+00	 1.5936873e-01	[ 1.1847949e+00]	 2.0287275e-01


.. parsed-literal::

      56	 1.1507657e+00	 1.5629288e-01	 1.1988666e+00	 1.5887984e-01	[ 1.1883603e+00]	 2.0584607e-01


.. parsed-literal::

      57	 1.1633061e+00	 1.5356031e-01	 1.2124112e+00	 1.5744836e-01	[ 1.1935281e+00]	 2.1721864e-01
      58	 1.1740967e+00	 1.5156712e-01	 1.2234447e+00	 1.5595063e-01	[ 1.2046116e+00]	 2.0166445e-01


.. parsed-literal::

      59	 1.1844305e+00	 1.5039775e-01	 1.2339593e+00	 1.5449266e-01	[ 1.2141483e+00]	 2.0583653e-01
      60	 1.1984056e+00	 1.4930267e-01	 1.2485228e+00	 1.5282338e-01	[ 1.2230246e+00]	 1.7334318e-01


.. parsed-literal::

      61	 1.2105758e+00	 1.4858480e-01	 1.2607741e+00	 1.5170928e-01	[ 1.2411820e+00]	 2.0562696e-01
      62	 1.2217894e+00	 1.4840150e-01	 1.2718514e+00	 1.5130968e-01	[ 1.2480096e+00]	 1.8788314e-01


.. parsed-literal::

      63	 1.2290244e+00	 1.4725521e-01	 1.2792639e+00	 1.5051520e-01	  1.2474685e+00 	 2.0166802e-01
      64	 1.2344750e+00	 1.4695993e-01	 1.2846279e+00	 1.5044766e-01	[ 1.2511842e+00]	 1.8531656e-01


.. parsed-literal::

      65	 1.2417873e+00	 1.4642506e-01	 1.2921851e+00	 1.5038739e-01	[ 1.2543649e+00]	 2.1850896e-01
      66	 1.2501145e+00	 1.4581128e-01	 1.3006836e+00	 1.4989658e-01	[ 1.2575620e+00]	 1.8676829e-01


.. parsed-literal::

      67	 1.2595296e+00	 1.4415152e-01	 1.3104091e+00	 1.4878247e-01	[ 1.2641206e+00]	 2.1664286e-01


.. parsed-literal::

      68	 1.2684234e+00	 1.4345068e-01	 1.3191587e+00	 1.4798539e-01	[ 1.2670778e+00]	 2.1349716e-01
      69	 1.2729285e+00	 1.4292039e-01	 1.3235720e+00	 1.4727742e-01	[ 1.2704244e+00]	 1.8644118e-01


.. parsed-literal::

      70	 1.2814071e+00	 1.4169609e-01	 1.3322093e+00	 1.4619051e-01	[ 1.2764436e+00]	 1.8194723e-01


.. parsed-literal::

      71	 1.2893281e+00	 1.4127787e-01	 1.3403958e+00	 1.4574504e-01	  1.2735420e+00 	 2.0217443e-01
      72	 1.2981376e+00	 1.4055956e-01	 1.3491396e+00	 1.4546265e-01	[ 1.2803353e+00]	 2.0157576e-01


.. parsed-literal::

      73	 1.3045681e+00	 1.4033737e-01	 1.3557459e+00	 1.4559156e-01	[ 1.2848070e+00]	 2.1669602e-01
      74	 1.3093714e+00	 1.4035703e-01	 1.3607736e+00	 1.4546051e-01	[ 1.2869917e+00]	 1.9992113e-01


.. parsed-literal::

      75	 1.3195578e+00	 1.3971438e-01	 1.3711806e+00	 1.4431952e-01	[ 1.2953554e+00]	 2.0887899e-01
      76	 1.3267524e+00	 1.3931469e-01	 1.3787583e+00	 1.4290003e-01	[ 1.3019485e+00]	 1.7928410e-01


.. parsed-literal::

      77	 1.3323682e+00	 1.3861943e-01	 1.3841350e+00	 1.4231055e-01	[ 1.3088143e+00]	 2.1051121e-01
      78	 1.3389304e+00	 1.3746523e-01	 1.3907357e+00	 1.4103187e-01	[ 1.3133554e+00]	 1.9185257e-01


.. parsed-literal::

      79	 1.3448047e+00	 1.3665397e-01	 1.3968685e+00	 1.4035195e-01	[ 1.3174249e+00]	 1.8304443e-01


.. parsed-literal::

      80	 1.3509195e+00	 1.3598796e-01	 1.4032853e+00	 1.3987225e-01	  1.3114268e+00 	 2.0395446e-01
      81	 1.3560083e+00	 1.3550520e-01	 1.4086777e+00	 1.3940317e-01	  1.3139686e+00 	 1.9914937e-01


.. parsed-literal::

      82	 1.3629008e+00	 1.3520638e-01	 1.4158707e+00	 1.3932202e-01	[ 1.3174295e+00]	 2.1070051e-01


.. parsed-literal::

      83	 1.3686301e+00	 1.3394694e-01	 1.4218920e+00	 1.3763875e-01	[ 1.3227287e+00]	 2.0413971e-01


.. parsed-literal::

      84	 1.3747709e+00	 1.3341824e-01	 1.4278682e+00	 1.3718029e-01	[ 1.3302123e+00]	 2.0147705e-01
      85	 1.3801269e+00	 1.3261033e-01	 1.4331840e+00	 1.3624546e-01	[ 1.3348993e+00]	 1.9561291e-01


.. parsed-literal::

      86	 1.3849670e+00	 1.3189124e-01	 1.4379996e+00	 1.3548914e-01	[ 1.3403785e+00]	 2.0817947e-01


.. parsed-literal::

      87	 1.3914895e+00	 1.3080178e-01	 1.4446307e+00	 1.3427256e-01	[ 1.3433043e+00]	 2.1617723e-01


.. parsed-literal::

      88	 1.3975926e+00	 1.3021168e-01	 1.4506409e+00	 1.3412861e-01	[ 1.3505565e+00]	 2.1214652e-01


.. parsed-literal::

      89	 1.4016394e+00	 1.3010499e-01	 1.4546173e+00	 1.3434054e-01	[ 1.3549295e+00]	 2.1913099e-01


.. parsed-literal::

      90	 1.4076910e+00	 1.2952549e-01	 1.4609311e+00	 1.3412295e-01	[ 1.3551161e+00]	 2.1803665e-01


.. parsed-literal::

      91	 1.4107809e+00	 1.2884288e-01	 1.4642243e+00	 1.3413248e-01	[ 1.3630915e+00]	 2.1455097e-01


.. parsed-literal::

      92	 1.4159631e+00	 1.2856911e-01	 1.4692815e+00	 1.3385839e-01	  1.3612906e+00 	 2.0961046e-01


.. parsed-literal::

      93	 1.4184414e+00	 1.2820450e-01	 1.4717674e+00	 1.3361807e-01	  1.3599041e+00 	 2.0390296e-01
      94	 1.4222869e+00	 1.2782029e-01	 1.4757445e+00	 1.3354141e-01	  1.3564909e+00 	 1.8125749e-01


.. parsed-literal::

      95	 1.4273473e+00	 1.2715950e-01	 1.4810197e+00	 1.3356868e-01	  1.3527626e+00 	 2.1419430e-01


.. parsed-literal::

      96	 1.4300477e+00	 1.2687251e-01	 1.4840947e+00	 1.3437484e-01	  1.3440159e+00 	 2.0438671e-01


.. parsed-literal::

      97	 1.4364482e+00	 1.2647277e-01	 1.4902722e+00	 1.3392117e-01	  1.3518325e+00 	 2.1470332e-01


.. parsed-literal::

      98	 1.4389936e+00	 1.2630635e-01	 1.4927735e+00	 1.3384057e-01	  1.3560359e+00 	 2.0537019e-01
      99	 1.4426831e+00	 1.2599227e-01	 1.4965338e+00	 1.3387501e-01	  1.3571886e+00 	 1.7477727e-01


.. parsed-literal::

     100	 1.4444950e+00	 1.2532710e-01	 1.4985739e+00	 1.3428290e-01	  1.3568728e+00 	 2.0852971e-01


.. parsed-literal::

     101	 1.4489892e+00	 1.2507216e-01	 1.5029790e+00	 1.3413680e-01	  1.3554691e+00 	 2.0711470e-01


.. parsed-literal::

     102	 1.4512748e+00	 1.2478038e-01	 1.5052983e+00	 1.3413384e-01	  1.3524152e+00 	 2.1016216e-01


.. parsed-literal::

     103	 1.4547407e+00	 1.2433416e-01	 1.5088884e+00	 1.3411297e-01	  1.3489511e+00 	 2.1413565e-01
     104	 1.4597599e+00	 1.2368936e-01	 1.5140348e+00	 1.3420989e-01	  1.3450284e+00 	 1.8549418e-01


.. parsed-literal::

     105	 1.4621287e+00	 1.2344038e-01	 1.5164978e+00	 1.3414163e-01	  1.3490132e+00 	 2.8932142e-01
     106	 1.4649305e+00	 1.2322468e-01	 1.5192982e+00	 1.3405723e-01	  1.3525325e+00 	 2.0363069e-01


.. parsed-literal::

     107	 1.4672718e+00	 1.2313423e-01	 1.5216954e+00	 1.3420386e-01	  1.3555560e+00 	 1.6304541e-01
     108	 1.4708001e+00	 1.2316817e-01	 1.5253083e+00	 1.3437686e-01	  1.3590740e+00 	 1.7144418e-01


.. parsed-literal::

     109	 1.4743686e+00	 1.2301073e-01	 1.5289630e+00	 1.3479328e-01	  1.3625239e+00 	 1.8737769e-01


.. parsed-literal::

     110	 1.4773436e+00	 1.2273953e-01	 1.5319083e+00	 1.3466117e-01	  1.3618205e+00 	 2.0995665e-01
     111	 1.4810939e+00	 1.2225609e-01	 1.5357213e+00	 1.3470147e-01	  1.3544677e+00 	 2.0343709e-01


.. parsed-literal::

     112	 1.4832517e+00	 1.2168106e-01	 1.5379455e+00	 1.3468087e-01	  1.3485493e+00 	 1.9923759e-01
     113	 1.4856045e+00	 1.2165024e-01	 1.5402554e+00	 1.3483201e-01	  1.3471475e+00 	 1.9651055e-01


.. parsed-literal::

     114	 1.4882025e+00	 1.2157303e-01	 1.5429377e+00	 1.3536580e-01	  1.3422740e+00 	 1.9979501e-01
     115	 1.4899059e+00	 1.2147525e-01	 1.5446572e+00	 1.3554770e-01	  1.3401727e+00 	 1.8924809e-01


.. parsed-literal::

     116	 1.4931486e+00	 1.2121146e-01	 1.5480341e+00	 1.3590803e-01	  1.3308043e+00 	 1.9987535e-01
     117	 1.4953176e+00	 1.2076638e-01	 1.5502300e+00	 1.3599750e-01	  1.3315488e+00 	 1.7304158e-01


.. parsed-literal::

     118	 1.4974709e+00	 1.2075280e-01	 1.5522507e+00	 1.3558113e-01	  1.3342084e+00 	 2.1813273e-01
     119	 1.4990836e+00	 1.2061582e-01	 1.5538741e+00	 1.3548185e-01	  1.3341438e+00 	 1.8533373e-01


.. parsed-literal::

     120	 1.5011590e+00	 1.2041409e-01	 1.5560516e+00	 1.3554131e-01	  1.3329188e+00 	 2.0712876e-01


.. parsed-literal::

     121	 1.5021126e+00	 1.2013571e-01	 1.5572533e+00	 1.3578998e-01	  1.3323604e+00 	 2.0562863e-01
     122	 1.5043736e+00	 1.2008741e-01	 1.5594534e+00	 1.3605002e-01	  1.3329684e+00 	 1.8726802e-01


.. parsed-literal::

     123	 1.5052337e+00	 1.2007103e-01	 1.5603050e+00	 1.3624259e-01	  1.3335930e+00 	 2.0531154e-01


.. parsed-literal::

     124	 1.5067818e+00	 1.2002727e-01	 1.5618565e+00	 1.3657487e-01	  1.3349589e+00 	 2.0444989e-01
     125	 1.5086929e+00	 1.1988433e-01	 1.5637860e+00	 1.3658474e-01	  1.3370268e+00 	 1.8753958e-01


.. parsed-literal::

     126	 1.5104980e+00	 1.1981498e-01	 1.5656642e+00	 1.3670187e-01	  1.3394798e+00 	 2.0418334e-01


.. parsed-literal::

     127	 1.5121709e+00	 1.1968520e-01	 1.5672830e+00	 1.3615160e-01	  1.3406665e+00 	 2.0870018e-01
     128	 1.5143218e+00	 1.1946148e-01	 1.5694665e+00	 1.3534512e-01	  1.3383108e+00 	 2.0041895e-01


.. parsed-literal::

     129	 1.5161059e+00	 1.1934503e-01	 1.5713218e+00	 1.3494506e-01	  1.3370854e+00 	 1.8350506e-01


.. parsed-literal::

     130	 1.5185031e+00	 1.1913089e-01	 1.5739189e+00	 1.3449236e-01	  1.3280968e+00 	 2.0854616e-01


.. parsed-literal::

     131	 1.5199771e+00	 1.1906168e-01	 1.5754493e+00	 1.3473497e-01	  1.3259338e+00 	 2.1491504e-01


.. parsed-literal::

     132	 1.5211751e+00	 1.1902598e-01	 1.5765825e+00	 1.3494763e-01	  1.3293774e+00 	 2.1148944e-01
     133	 1.5227863e+00	 1.1891675e-01	 1.5781492e+00	 1.3497297e-01	  1.3308540e+00 	 1.7732930e-01


.. parsed-literal::

     134	 1.5243579e+00	 1.1868234e-01	 1.5797541e+00	 1.3470609e-01	  1.3324696e+00 	 1.8099213e-01


.. parsed-literal::

     135	 1.5259940e+00	 1.1854337e-01	 1.5814048e+00	 1.3415977e-01	  1.3323726e+00 	 2.0528388e-01


.. parsed-literal::

     136	 1.5274384e+00	 1.1838408e-01	 1.5828843e+00	 1.3365454e-01	  1.3305296e+00 	 2.1370769e-01


.. parsed-literal::

     137	 1.5289392e+00	 1.1822379e-01	 1.5844847e+00	 1.3322215e-01	  1.3277302e+00 	 2.0863247e-01


.. parsed-literal::

     138	 1.5299123e+00	 1.1806137e-01	 1.5855700e+00	 1.3301171e-01	  1.3257951e+00 	 2.0808125e-01


.. parsed-literal::

     139	 1.5312530e+00	 1.1802163e-01	 1.5868846e+00	 1.3326241e-01	  1.3250842e+00 	 2.0716643e-01
     140	 1.5325392e+00	 1.1791444e-01	 1.5882013e+00	 1.3369157e-01	  1.3224979e+00 	 1.8047905e-01


.. parsed-literal::

     141	 1.5334931e+00	 1.1782642e-01	 1.5891623e+00	 1.3390942e-01	  1.3221702e+00 	 1.7409945e-01
     142	 1.5352184e+00	 1.1764516e-01	 1.5909686e+00	 1.3421937e-01	  1.3228740e+00 	 1.7629075e-01


.. parsed-literal::

     143	 1.5369097e+00	 1.1753164e-01	 1.5926378e+00	 1.3432671e-01	  1.3308430e+00 	 2.1748066e-01


.. parsed-literal::

     144	 1.5378728e+00	 1.1742977e-01	 1.5935666e+00	 1.3396249e-01	  1.3318116e+00 	 2.1811104e-01
     145	 1.5392043e+00	 1.1728169e-01	 1.5949191e+00	 1.3358628e-01	  1.3342561e+00 	 1.7385197e-01


.. parsed-literal::

     146	 1.5403956e+00	 1.1708306e-01	 1.5962098e+00	 1.3344202e-01	  1.3334198e+00 	 1.9228673e-01
     147	 1.5417970e+00	 1.1695324e-01	 1.5977069e+00	 1.3358753e-01	  1.3300903e+00 	 2.0220017e-01


.. parsed-literal::

     148	 1.5432137e+00	 1.1684119e-01	 1.5992311e+00	 1.3396668e-01	  1.3290045e+00 	 1.7621422e-01


.. parsed-literal::

     149	 1.5443388e+00	 1.1679367e-01	 1.6003868e+00	 1.3412067e-01	  1.3277772e+00 	 2.0290017e-01


.. parsed-literal::

     150	 1.5458498e+00	 1.1665720e-01	 1.6019021e+00	 1.3396764e-01	  1.3259671e+00 	 2.1295452e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.24 s, total: 2min 3s
    Wall time: 30.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fefd0ff11b0>



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
    CPU times: user 2.01 s, sys: 35 ms, total: 2.05 s
    Wall time: 607 ms


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

