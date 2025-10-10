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
       1	-3.2821249e-01	 3.1594011e-01	-3.1846849e-01	 3.3827124e-01	[-3.6377004e-01]	 4.5617151e-01


.. parsed-literal::

       2	-2.5528723e-01	 3.0421021e-01	-2.3034593e-01	 3.2680747e-01	[-3.0253227e-01]	 2.2685623e-01


.. parsed-literal::

       3	-2.1267768e-01	 2.8555851e-01	-1.7192826e-01	 3.0632040e-01	[-2.6469373e-01]	 2.7292871e-01
       4	-1.8375912e-01	 2.6129234e-01	-1.4384856e-01	 2.8156236e-01	 -2.6948653e-01 	 1.7378640e-01


.. parsed-literal::

       5	-9.4582500e-02	 2.5379954e-01	-5.7894374e-02	 2.7282901e-01	[-1.5632234e-01]	 2.0154977e-01


.. parsed-literal::

       6	-5.7222642e-02	 2.4747342e-01	-2.3724188e-02	 2.6976417e-01	[-1.0826161e-01]	 2.0638180e-01
       7	-3.9641906e-02	 2.4514874e-01	-1.3665457e-02	 2.6641581e-01	[-1.0059483e-01]	 1.9452524e-01


.. parsed-literal::

       8	-2.5611344e-02	 2.4290607e-01	-4.1646150e-03	 2.6346743e-01	[-8.9963689e-02]	 1.9515443e-01


.. parsed-literal::

       9	-1.0491339e-02	 2.4024426e-01	 7.3850146e-03	 2.6031643e-01	[-7.9181291e-02]	 2.1131492e-01
      10	-2.1264296e-03	 2.3890050e-01	 1.3497601e-02	 2.5736352e-01	[-6.3292905e-02]	 1.8166757e-01


.. parsed-literal::

      11	 3.7247822e-03	 2.3778906e-01	 1.8327601e-02	 2.5660367e-01	 -6.7151484e-02 	 1.9902372e-01
      12	 6.6364003e-03	 2.3731788e-01	 2.1018233e-02	 2.5610048e-01	[-6.0997740e-02]	 1.9992638e-01


.. parsed-literal::

      13	 9.3969113e-03	 2.3683560e-01	 2.3463758e-02	 2.5543321e-01	[-5.7010657e-02]	 2.1381140e-01


.. parsed-literal::

      14	 1.4325933e-02	 2.3581506e-01	 2.8610823e-02	 2.5413725e-01	[-4.8927328e-02]	 2.0239186e-01


.. parsed-literal::

      15	 1.0357593e-01	 2.2162266e-01	 1.2299045e-01	 2.3964561e-01	[ 7.3433400e-02]	 3.2662058e-01


.. parsed-literal::

      16	 1.9178852e-01	 2.1782467e-01	 2.1419758e-01	 2.3428755e-01	[ 1.5995104e-01]	 2.1046686e-01
      17	 2.7964030e-01	 2.1231181e-01	 3.0968757e-01	 2.2806426e-01	[ 2.2386079e-01]	 1.9936728e-01


.. parsed-literal::

      18	 3.2065260e-01	 2.1158341e-01	 3.5212639e-01	 2.2781590e-01	[ 2.8124264e-01]	 2.0550132e-01
      19	 3.6765338e-01	 2.0699948e-01	 3.9912542e-01	 2.2208082e-01	[ 3.4333803e-01]	 1.8238592e-01


.. parsed-literal::

      20	 4.3104993e-01	 2.0123989e-01	 4.6347923e-01	 2.1454230e-01	[ 4.1086057e-01]	 2.0073271e-01


.. parsed-literal::

      21	 5.3006466e-01	 2.0109049e-01	 5.6520361e-01	 2.1453985e-01	[ 5.1128217e-01]	 2.1185493e-01


.. parsed-literal::

      22	 5.6254525e-01	 2.0392913e-01	 6.0073366e-01	 2.2066246e-01	[ 5.5742386e-01]	 2.1092677e-01
      23	 6.0952906e-01	 1.9885707e-01	 6.4662580e-01	 2.1320270e-01	[ 6.1479177e-01]	 1.7767310e-01


.. parsed-literal::

      24	 6.4106120e-01	 1.9568630e-01	 6.7783635e-01	 2.0883724e-01	[ 6.4553552e-01]	 1.8086147e-01


.. parsed-literal::

      25	 6.6760947e-01	 1.9799589e-01	 7.0306362e-01	 2.1336654e-01	[ 6.5787934e-01]	 2.0924592e-01
      26	 6.9985539e-01	 1.9556862e-01	 7.3585955e-01	 2.0936455e-01	[ 6.9909083e-01]	 1.8910074e-01


.. parsed-literal::

      27	 7.1049414e-01	 1.9622350e-01	 7.4717223e-01	 2.1008344e-01	[ 7.0517123e-01]	 2.1326494e-01
      28	 7.2325867e-01	 1.9490787e-01	 7.5976591e-01	 2.0913550e-01	[ 7.1943669e-01]	 1.7951560e-01


.. parsed-literal::

      29	 7.5054801e-01	 1.9496077e-01	 7.8749112e-01	 2.1187578e-01	[ 7.4540967e-01]	 2.0771408e-01
      30	 7.7636111e-01	 1.9407555e-01	 8.1334394e-01	 2.1145301e-01	[ 7.7331388e-01]	 1.8790030e-01


.. parsed-literal::

      31	 8.1151802e-01	 1.9376486e-01	 8.4982410e-01	 2.1327057e-01	[ 8.0491121e-01]	 2.0798635e-01


.. parsed-literal::

      32	 8.5013342e-01	 1.8985754e-01	 8.8950556e-01	 2.1146724e-01	[ 8.3598549e-01]	 2.1157002e-01


.. parsed-literal::

      33	 8.7029049e-01	 1.8518918e-01	 9.1057662e-01	 2.0326609e-01	[ 8.5529973e-01]	 2.1025062e-01
      34	 8.8483646e-01	 1.8233652e-01	 9.2529265e-01	 2.0235515e-01	[ 8.6518418e-01]	 1.9441342e-01


.. parsed-literal::

      35	 8.9365596e-01	 1.8146673e-01	 9.3433849e-01	 2.0193904e-01	[ 8.7130284e-01]	 1.9960070e-01


.. parsed-literal::

      36	 9.0980914e-01	 1.7877060e-01	 9.5077294e-01	 1.9882911e-01	[ 8.8658654e-01]	 2.0275354e-01
      37	 9.3440663e-01	 1.7690384e-01	 9.7615838e-01	 1.9633530e-01	[ 9.1713999e-01]	 2.0285606e-01


.. parsed-literal::

      38	 9.5980415e-01	 1.7492441e-01	 1.0029557e+00	 1.9380477e-01	[ 9.3795157e-01]	 2.0268679e-01


.. parsed-literal::

      39	 9.7553793e-01	 1.7373891e-01	 1.0192281e+00	 1.9246768e-01	[ 9.6110639e-01]	 2.0894456e-01


.. parsed-literal::

      40	 9.9041076e-01	 1.7310584e-01	 1.0343433e+00	 1.9180268e-01	[ 9.7519832e-01]	 2.1683097e-01
      41	 1.0115978e+00	 1.7087201e-01	 1.0565635e+00	 1.8883400e-01	[ 1.0008020e+00]	 1.9762158e-01


.. parsed-literal::

      42	 1.0224845e+00	 1.7024255e-01	 1.0678597e+00	 1.8876123e-01	[ 1.0053491e+00]	 2.0572972e-01


.. parsed-literal::

      43	 1.0306805e+00	 1.6894905e-01	 1.0759587e+00	 1.8728707e-01	[ 1.0143295e+00]	 2.0768619e-01


.. parsed-literal::

      44	 1.0456249e+00	 1.6579140e-01	 1.0912924e+00	 1.8379743e-01	[ 1.0326387e+00]	 2.0648861e-01


.. parsed-literal::

      45	 1.0551079e+00	 1.6490365e-01	 1.1004890e+00	 1.8300033e-01	[ 1.0387209e+00]	 2.1434379e-01


.. parsed-literal::

      46	 1.0724904e+00	 1.6287606e-01	 1.1179060e+00	 1.8136324e-01	[ 1.0485811e+00]	 2.1268749e-01


.. parsed-literal::

      47	 1.0858770e+00	 1.6026839e-01	 1.1313937e+00	 1.7964039e-01	[ 1.0581329e+00]	 2.1162105e-01
      48	 1.1009738e+00	 1.5651235e-01	 1.1468572e+00	 1.7675366e-01	[ 1.0726616e+00]	 1.7299175e-01


.. parsed-literal::

      49	 1.1113147e+00	 1.5506928e-01	 1.1577899e+00	 1.7621345e-01	[ 1.0793101e+00]	 2.0000124e-01


.. parsed-literal::

      50	 1.1175080e+00	 1.5462106e-01	 1.1640725e+00	 1.7542898e-01	[ 1.0873592e+00]	 2.0667577e-01
      51	 1.1285874e+00	 1.5330209e-01	 1.1756904e+00	 1.7395437e-01	[ 1.0983517e+00]	 1.9840956e-01


.. parsed-literal::

      52	 1.1401532e+00	 1.5165363e-01	 1.1876865e+00	 1.7228366e-01	[ 1.1108100e+00]	 2.1257138e-01
      53	 1.1527666e+00	 1.5031652e-01	 1.2007083e+00	 1.7143241e-01	[ 1.1255939e+00]	 1.9683719e-01


.. parsed-literal::

      54	 1.1621338e+00	 1.4941696e-01	 1.2104090e+00	 1.7101237e-01	[ 1.1356379e+00]	 2.0841193e-01


.. parsed-literal::

      55	 1.1747870e+00	 1.4822465e-01	 1.2234689e+00	 1.6971381e-01	[ 1.1479484e+00]	 2.0827293e-01
      56	 1.1835359e+00	 1.4840631e-01	 1.2320752e+00	 1.7080994e-01	[ 1.1530917e+00]	 1.7620826e-01


.. parsed-literal::

      57	 1.1908815e+00	 1.4790546e-01	 1.2393312e+00	 1.6976547e-01	[ 1.1623813e+00]	 2.0839214e-01


.. parsed-literal::

      58	 1.1986218e+00	 1.4694747e-01	 1.2473849e+00	 1.6818098e-01	[ 1.1696472e+00]	 2.8641891e-01
      59	 1.2078429e+00	 1.4607947e-01	 1.2568844e+00	 1.6708126e-01	[ 1.1753251e+00]	 1.9954133e-01


.. parsed-literal::

      60	 1.2166745e+00	 1.4556786e-01	 1.2662050e+00	 1.6667350e-01	[ 1.1802555e+00]	 1.9798899e-01


.. parsed-literal::

      61	 1.2259155e+00	 1.4501381e-01	 1.2754289e+00	 1.6655039e-01	[ 1.1870837e+00]	 2.0893741e-01


.. parsed-literal::

      62	 1.2350175e+00	 1.4446063e-01	 1.2847510e+00	 1.6668000e-01	[ 1.1913459e+00]	 2.1432233e-01
      63	 1.2431630e+00	 1.4411527e-01	 1.2933058e+00	 1.6720198e-01	[ 1.1918122e+00]	 1.9738293e-01


.. parsed-literal::

      64	 1.2506615e+00	 1.4392913e-01	 1.3009469e+00	 1.6770909e-01	[ 1.1948605e+00]	 2.0558667e-01


.. parsed-literal::

      65	 1.2559329e+00	 1.4366984e-01	 1.3061508e+00	 1.6761881e-01	[ 1.1973612e+00]	 2.0629525e-01


.. parsed-literal::

      66	 1.2648131e+00	 1.4280126e-01	 1.3153599e+00	 1.6673489e-01	[ 1.1993292e+00]	 2.1340466e-01


.. parsed-literal::

      67	 1.2708405e+00	 1.4140514e-01	 1.3215687e+00	 1.6543517e-01	  1.1965441e+00 	 2.1501064e-01
      68	 1.2769274e+00	 1.4115839e-01	 1.3275628e+00	 1.6488004e-01	[ 1.2076564e+00]	 1.7069221e-01


.. parsed-literal::

      69	 1.2825857e+00	 1.4028615e-01	 1.3334948e+00	 1.6345186e-01	[ 1.2133022e+00]	 2.1783209e-01
      70	 1.2871891e+00	 1.3970831e-01	 1.3381580e+00	 1.6266728e-01	[ 1.2206532e+00]	 1.8856144e-01


.. parsed-literal::

      71	 1.2925580e+00	 1.3901802e-01	 1.3435587e+00	 1.6174729e-01	[ 1.2243735e+00]	 2.1520042e-01


.. parsed-literal::

      72	 1.2989965e+00	 1.3830258e-01	 1.3500579e+00	 1.6079678e-01	[ 1.2279986e+00]	 2.0161128e-01


.. parsed-literal::

      73	 1.3043384e+00	 1.3799306e-01	 1.3553605e+00	 1.6044306e-01	[ 1.2322919e+00]	 2.2100282e-01
      74	 1.3111696e+00	 1.3768342e-01	 1.3623067e+00	 1.5986415e-01	[ 1.2335376e+00]	 2.0074940e-01


.. parsed-literal::

      75	 1.3168919e+00	 1.3765633e-01	 1.3681729e+00	 1.5972758e-01	[ 1.2359139e+00]	 1.7910695e-01


.. parsed-literal::

      76	 1.3225437e+00	 1.3769654e-01	 1.3740020e+00	 1.5991695e-01	[ 1.2381837e+00]	 2.1324587e-01


.. parsed-literal::

      77	 1.3288479e+00	 1.3708434e-01	 1.3807306e+00	 1.5923019e-01	[ 1.2409093e+00]	 2.1583843e-01
      78	 1.3346557e+00	 1.3670700e-01	 1.3866586e+00	 1.5929315e-01	[ 1.2441918e+00]	 1.9729400e-01


.. parsed-literal::

      79	 1.3391253e+00	 1.3644225e-01	 1.3911180e+00	 1.5915389e-01	[ 1.2472628e+00]	 1.9767833e-01
      80	 1.3449776e+00	 1.3579470e-01	 1.3971193e+00	 1.5851589e-01	  1.2468660e+00 	 1.8647432e-01


.. parsed-literal::

      81	 1.3499237e+00	 1.3583179e-01	 1.4021953e+00	 1.5819891e-01	  1.2467585e+00 	 2.1061945e-01


.. parsed-literal::

      82	 1.3541826e+00	 1.3545887e-01	 1.4063907e+00	 1.5762086e-01	  1.2457218e+00 	 2.0319438e-01


.. parsed-literal::

      83	 1.3590583e+00	 1.3509042e-01	 1.4113584e+00	 1.5705102e-01	  1.2386721e+00 	 2.0255494e-01
      84	 1.3628957e+00	 1.3477308e-01	 1.4152772e+00	 1.5654098e-01	  1.2341251e+00 	 1.7505169e-01


.. parsed-literal::

      85	 1.3670314e+00	 1.3463913e-01	 1.4195202e+00	 1.5635509e-01	  1.2328960e+00 	 2.0845413e-01


.. parsed-literal::

      86	 1.3712748e+00	 1.3436914e-01	 1.4238675e+00	 1.5612026e-01	  1.2288152e+00 	 2.0496154e-01


.. parsed-literal::

      87	 1.3745766e+00	 1.3401231e-01	 1.4272280e+00	 1.5566803e-01	  1.2288094e+00 	 2.1772122e-01


.. parsed-literal::

      88	 1.3794005e+00	 1.3334179e-01	 1.4321000e+00	 1.5475270e-01	  1.2239138e+00 	 2.1112132e-01
      89	 1.3819095e+00	 1.3261876e-01	 1.4349175e+00	 1.5406936e-01	  1.2127182e+00 	 1.8051958e-01


.. parsed-literal::

      90	 1.3872974e+00	 1.3205485e-01	 1.4401504e+00	 1.5328861e-01	  1.2177167e+00 	 2.0371485e-01
      91	 1.3901158e+00	 1.3171183e-01	 1.4429405e+00	 1.5293732e-01	  1.2215224e+00 	 1.8741870e-01


.. parsed-literal::

      92	 1.3938351e+00	 1.3121194e-01	 1.4467647e+00	 1.5253821e-01	  1.2219061e+00 	 2.0268726e-01


.. parsed-literal::

      93	 1.3990788e+00	 1.3042579e-01	 1.4521887e+00	 1.5217806e-01	  1.2217650e+00 	 2.1879554e-01


.. parsed-literal::

      94	 1.4032621e+00	 1.2977284e-01	 1.4566553e+00	 1.5189222e-01	  1.2145163e+00 	 2.0354080e-01
      95	 1.4070536e+00	 1.2979037e-01	 1.4602515e+00	 1.5205028e-01	  1.2144535e+00 	 1.9782352e-01


.. parsed-literal::

      96	 1.4097812e+00	 1.2960924e-01	 1.4629348e+00	 1.5208868e-01	  1.2082172e+00 	 1.9091272e-01
      97	 1.4126683e+00	 1.2928957e-01	 1.4658538e+00	 1.5190237e-01	  1.1980610e+00 	 1.9508505e-01


.. parsed-literal::

      98	 1.4155022e+00	 1.2916411e-01	 1.4687899e+00	 1.5220907e-01	  1.1840927e+00 	 1.8828821e-01


.. parsed-literal::

      99	 1.4176515e+00	 1.2881858e-01	 1.4709346e+00	 1.5188834e-01	  1.1844761e+00 	 2.0596719e-01


.. parsed-literal::

     100	 1.4204069e+00	 1.2843484e-01	 1.4736801e+00	 1.5143142e-01	  1.1851697e+00 	 2.0788264e-01


.. parsed-literal::

     101	 1.4237239e+00	 1.2797607e-01	 1.4770368e+00	 1.5113565e-01	  1.1859868e+00 	 2.0870972e-01


.. parsed-literal::

     102	 1.4272333e+00	 1.2737041e-01	 1.4807101e+00	 1.5072124e-01	  1.1662120e+00 	 2.0856428e-01


.. parsed-literal::

     103	 1.4313039e+00	 1.2727293e-01	 1.4846903e+00	 1.5062307e-01	  1.1774717e+00 	 2.0392799e-01


.. parsed-literal::

     104	 1.4332062e+00	 1.2722521e-01	 1.4865431e+00	 1.5045264e-01	  1.1770922e+00 	 2.1504545e-01
     105	 1.4360003e+00	 1.2700596e-01	 1.4894038e+00	 1.5003337e-01	  1.1641467e+00 	 1.9687152e-01


.. parsed-literal::

     106	 1.4384441e+00	 1.2650396e-01	 1.4920859e+00	 1.4877703e-01	  1.1428297e+00 	 2.1028948e-01


.. parsed-literal::

     107	 1.4417207e+00	 1.2631715e-01	 1.4954243e+00	 1.4862390e-01	  1.1273762e+00 	 2.1890736e-01
     108	 1.4436896e+00	 1.2614142e-01	 1.4974597e+00	 1.4841548e-01	  1.1170505e+00 	 1.9397998e-01


.. parsed-literal::

     109	 1.4462303e+00	 1.2592154e-01	 1.5000997e+00	 1.4809581e-01	  1.1052859e+00 	 2.0489407e-01


.. parsed-literal::

     110	 1.4495411e+00	 1.2524957e-01	 1.5036358e+00	 1.4715086e-01	  1.0788611e+00 	 2.0577431e-01


.. parsed-literal::

     111	 1.4528681e+00	 1.2493749e-01	 1.5070947e+00	 1.4676122e-01	  1.0522853e+00 	 2.1773434e-01
     112	 1.4549728e+00	 1.2480732e-01	 1.5091627e+00	 1.4658421e-01	  1.0494148e+00 	 1.9418192e-01


.. parsed-literal::

     113	 1.4574160e+00	 1.2458636e-01	 1.5116473e+00	 1.4637207e-01	  1.0361007e+00 	 1.9631529e-01


.. parsed-literal::

     114	 1.4594695e+00	 1.2437001e-01	 1.5137526e+00	 1.4612140e-01	  1.0229834e+00 	 2.1296787e-01


.. parsed-literal::

     115	 1.4616391e+00	 1.2422186e-01	 1.5159925e+00	 1.4598050e-01	  1.0103574e+00 	 2.0957351e-01


.. parsed-literal::

     116	 1.4641902e+00	 1.2403427e-01	 1.5186580e+00	 1.4581458e-01	  1.0055346e+00 	 2.0847297e-01


.. parsed-literal::

     117	 1.4660388e+00	 1.2392741e-01	 1.5204156e+00	 1.4568160e-01	  1.0045831e+00 	 2.1459770e-01
     118	 1.4674139e+00	 1.2382406e-01	 1.5217235e+00	 1.4558922e-01	  1.0123909e+00 	 1.9629073e-01


.. parsed-literal::

     119	 1.4690565e+00	 1.2356201e-01	 1.5232984e+00	 1.4527654e-01	  1.0239799e+00 	 2.1136737e-01
     120	 1.4703831e+00	 1.2338737e-01	 1.5246132e+00	 1.4513712e-01	  1.0246009e+00 	 2.0363688e-01


.. parsed-literal::

     121	 1.4729518e+00	 1.2300622e-01	 1.5272127e+00	 1.4471897e-01	  1.0265601e+00 	 1.8200612e-01
     122	 1.4758891e+00	 1.2284154e-01	 1.5302460e+00	 1.4474316e-01	  1.0128367e+00 	 1.9562030e-01


.. parsed-literal::

     123	 1.4786786e+00	 1.2268788e-01	 1.5331201e+00	 1.4466352e-01	  1.0076959e+00 	 2.0226216e-01


.. parsed-literal::

     124	 1.4805747e+00	 1.2269938e-01	 1.5350284e+00	 1.4471290e-01	  9.9445781e-01 	 2.1547198e-01
     125	 1.4823524e+00	 1.2278336e-01	 1.5368144e+00	 1.4487570e-01	  9.8452681e-01 	 1.9990087e-01


.. parsed-literal::

     126	 1.4846933e+00	 1.2276425e-01	 1.5391923e+00	 1.4492110e-01	  9.5307467e-01 	 1.9813871e-01
     127	 1.4869334e+00	 1.2266934e-01	 1.5414639e+00	 1.4481703e-01	  9.3081938e-01 	 1.8657804e-01


.. parsed-literal::

     128	 1.4886566e+00	 1.2255212e-01	 1.5431674e+00	 1.4474436e-01	  9.2774448e-01 	 1.9757605e-01


.. parsed-literal::

     129	 1.4908094e+00	 1.2229140e-01	 1.5453271e+00	 1.4439918e-01	  9.1437207e-01 	 2.1703219e-01
     130	 1.4922177e+00	 1.2220441e-01	 1.5468195e+00	 1.4422634e-01	  8.7691825e-01 	 1.8060112e-01


.. parsed-literal::

     131	 1.4943632e+00	 1.2217228e-01	 1.5489716e+00	 1.4423286e-01	  8.6903228e-01 	 1.9304848e-01
     132	 1.4961519e+00	 1.2213880e-01	 1.5508434e+00	 1.4418050e-01	  8.4720815e-01 	 2.0000386e-01


.. parsed-literal::

     133	 1.4978283e+00	 1.2213386e-01	 1.5526020e+00	 1.4421359e-01	  8.2595530e-01 	 2.0063138e-01


.. parsed-literal::

     134	 1.4997971e+00	 1.2199843e-01	 1.5547050e+00	 1.4388446e-01	  8.2018409e-01 	 3.1935692e-01
     135	 1.5021029e+00	 1.2205293e-01	 1.5570562e+00	 1.4400594e-01	  7.9627336e-01 	 1.9694352e-01


.. parsed-literal::

     136	 1.5032851e+00	 1.2200204e-01	 1.5581805e+00	 1.4387226e-01	  8.0578823e-01 	 2.0632577e-01
     137	 1.5055137e+00	 1.2185174e-01	 1.5603600e+00	 1.4341467e-01	  8.1904054e-01 	 1.6684246e-01


.. parsed-literal::

     138	 1.5072209e+00	 1.2187670e-01	 1.5621065e+00	 1.4308692e-01	  8.3193326e-01 	 2.0209837e-01
     139	 1.5090670e+00	 1.2179427e-01	 1.5639322e+00	 1.4278638e-01	  8.2230131e-01 	 1.7754555e-01


.. parsed-literal::

     140	 1.5108184e+00	 1.2178979e-01	 1.5657337e+00	 1.4257181e-01	  8.0321770e-01 	 1.9711876e-01


.. parsed-literal::

     141	 1.5123166e+00	 1.2186819e-01	 1.5672922e+00	 1.4246363e-01	  7.8653987e-01 	 2.1143627e-01


.. parsed-literal::

     142	 1.5145436e+00	 1.2213760e-01	 1.5696461e+00	 1.4245147e-01	  7.4716757e-01 	 2.1255422e-01


.. parsed-literal::

     143	 1.5154179e+00	 1.2262980e-01	 1.5706425e+00	 1.4269871e-01	  7.2316329e-01 	 2.1281886e-01
     144	 1.5171808e+00	 1.2235818e-01	 1.5722837e+00	 1.4258567e-01	  7.3598024e-01 	 1.7722392e-01


.. parsed-literal::

     145	 1.5178424e+00	 1.2228816e-01	 1.5729159e+00	 1.4256797e-01	  7.3579216e-01 	 1.7766833e-01
     146	 1.5188522e+00	 1.2220501e-01	 1.5739182e+00	 1.4254419e-01	  7.2764799e-01 	 1.7955351e-01


.. parsed-literal::

     147	 1.5204191e+00	 1.2211630e-01	 1.5755201e+00	 1.4260762e-01	  6.9266909e-01 	 2.0587182e-01


.. parsed-literal::

     148	 1.5214012e+00	 1.2206434e-01	 1.5766723e+00	 1.4232969e-01	  6.6120675e-01 	 2.0441198e-01
     149	 1.5227996e+00	 1.2203142e-01	 1.5780021e+00	 1.4238905e-01	  6.5851896e-01 	 2.0060778e-01


.. parsed-literal::

     150	 1.5235548e+00	 1.2201655e-01	 1.5787774e+00	 1.4236521e-01	  6.5202670e-01 	 1.8297768e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.02 s, total: 2min 3s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1734f52aa0>



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
    CPU times: user 1.73 s, sys: 46 ms, total: 1.78 s
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

