GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

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
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

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
       1	-3.3495405e-01	 3.1775106e-01	-3.2471318e-01	 3.2836922e-01	[-3.4547300e-01]	 4.9193335e-01


.. parsed-literal::

       2	-2.6648745e-01	 3.0813637e-01	-2.4252680e-01	 3.1884889e-01	[-2.7633783e-01]	 2.5091577e-01


.. parsed-literal::

       3	-2.2165712e-01	 2.8767961e-01	-1.7707393e-01	 2.9829306e-01	[-2.2164359e-01]	 2.9809308e-01


.. parsed-literal::

       4	-1.8747086e-01	 2.7079454e-01	-1.3726187e-01	 2.8403130e-01	[-2.0711072e-01]	 3.2284594e-01
       5	-1.2516775e-01	 2.5464939e-01	-8.6124925e-02	 2.6936318e-01	[-1.6992960e-01]	 2.0906377e-01


.. parsed-literal::

       6	-6.0010516e-02	 2.4870460e-01	-2.8397013e-02	 2.6287298e-01	[-8.8577123e-02]	 2.1131754e-01


.. parsed-literal::

       7	-3.7903217e-02	 2.4511835e-01	-1.3090191e-02	 2.5861907e-01	[-7.1310135e-02]	 2.0971680e-01


.. parsed-literal::

       8	-2.3486602e-02	 2.4273605e-01	-3.8042883e-03	 2.5579920e-01	[-6.1918505e-02]	 2.0639324e-01


.. parsed-literal::

       9	-1.6148360e-02	 2.4148649e-01	 2.0581134e-03	 2.5435251e-01	[-5.0853727e-02]	 2.0985150e-01


.. parsed-literal::

      10	-8.7105572e-03	 2.4011105e-01	 8.1235685e-03	 2.5278223e-01	[-4.4350459e-02]	 2.0884991e-01
      11	-2.5455904e-04	 2.3857079e-01	 1.5257487e-02	 2.5078114e-01	[-3.5490333e-02]	 1.9925070e-01


.. parsed-literal::

      12	 4.7772871e-03	 2.3781666e-01	 1.9609983e-02	 2.5021588e-01	[-3.4528058e-02]	 1.9774461e-01
      13	 7.9652531e-03	 2.3713372e-01	 2.2927123e-02	 2.4944845e-01	[-3.0589865e-02]	 1.9603586e-01


.. parsed-literal::

      14	 3.4947074e-02	 2.3220732e-01	 5.1346524e-02	 2.4431211e-01	[-3.3088033e-04]	 2.0380831e-01


.. parsed-literal::

      15	 9.5059055e-02	 2.2434550e-01	 1.1584920e-01	 2.3395993e-01	[ 8.3719120e-02]	 3.3385229e-01
      16	 1.3458051e-01	 2.2161046e-01	 1.5610738e-01	 2.3000037e-01	[ 1.2627317e-01]	 1.9859934e-01


.. parsed-literal::

      17	 1.8930696e-01	 2.1763767e-01	 2.1438234e-01	 2.2716880e-01	[ 1.7228143e-01]	 1.9834518e-01


.. parsed-literal::

      18	 2.5071690e-01	 2.1248083e-01	 2.8001372e-01	 2.2325896e-01	[ 2.3527843e-01]	 2.1065521e-01


.. parsed-literal::

      19	 3.1685152e-01	 2.1325506e-01	 3.5049578e-01	 2.2698999e-01	[ 3.0643824e-01]	 2.0833135e-01


.. parsed-literal::

      20	 3.6609244e-01	 2.0483627e-01	 3.9963351e-01	 2.1519867e-01	[ 3.7057290e-01]	 2.1666050e-01
      21	 4.4362590e-01	 1.9898684e-01	 4.7830187e-01	 2.1020578e-01	[ 4.3729545e-01]	 1.9565082e-01


.. parsed-literal::

      22	 5.0518280e-01	 1.9534514e-01	 5.4259005e-01	 2.0786238e-01	[ 4.8691789e-01]	 1.9790506e-01


.. parsed-literal::

      23	 5.9602256e-01	 1.8952158e-01	 6.3650408e-01	 2.0039417e-01	[ 5.8794482e-01]	 2.0274425e-01


.. parsed-literal::

      24	 6.3791110e-01	 1.8674000e-01	 6.8238522e-01	 1.9175407e-01	[ 6.3896373e-01]	 2.1640134e-01
      25	 6.7298842e-01	 1.7736530e-01	 7.1242957e-01	 1.8635884e-01	[ 6.7897632e-01]	 2.0579123e-01


.. parsed-literal::

      26	 7.2359887e-01	 1.7725756e-01	 7.6385264e-01	 1.8646042e-01	[ 7.2217523e-01]	 2.0940638e-01


.. parsed-literal::

      27	 7.5673796e-01	 1.7647506e-01	 7.9739767e-01	 1.8600872e-01	[ 7.5639786e-01]	 3.3415103e-01


.. parsed-literal::

      28	 7.8678767e-01	 1.7647801e-01	 8.2843266e-01	 1.8545445e-01	[ 7.8636126e-01]	 2.0934677e-01


.. parsed-literal::

      29	 8.1811140e-01	 1.7588698e-01	 8.6037933e-01	 1.8546029e-01	[ 8.1848724e-01]	 2.1176887e-01


.. parsed-literal::

      30	 8.4323477e-01	 1.7252013e-01	 8.8455484e-01	 1.8130961e-01	[ 8.4214222e-01]	 2.1472740e-01


.. parsed-literal::

      31	 8.6153847e-01	 1.7086445e-01	 9.0254306e-01	 1.7988814e-01	[ 8.6551720e-01]	 2.0949554e-01
      32	 8.8811486e-01	 1.6821840e-01	 9.3002631e-01	 1.7785230e-01	[ 8.9154387e-01]	 1.9668055e-01


.. parsed-literal::

      33	 9.1289983e-01	 1.6720900e-01	 9.5647301e-01	 1.7524423e-01	[ 9.1335143e-01]	 1.9904780e-01
      34	 9.2794024e-01	 1.6511663e-01	 9.7264081e-01	 1.7430354e-01	[ 9.3509063e-01]	 1.8147755e-01


.. parsed-literal::

      35	 9.3962646e-01	 1.6302845e-01	 9.8464369e-01	 1.7354814e-01	[ 9.4532673e-01]	 1.8337274e-01


.. parsed-literal::

      36	 9.5968112e-01	 1.6086543e-01	 1.0052320e+00	 1.7314188e-01	[ 9.6450755e-01]	 2.0946407e-01
      37	 9.8312787e-01	 1.6067713e-01	 1.0295460e+00	 1.7368956e-01	[ 9.8424618e-01]	 1.9677019e-01


.. parsed-literal::

      38	 9.9895809e-01	 1.5935653e-01	 1.0463481e+00	 1.7429051e-01	[ 9.9987646e-01]	 1.7300653e-01


.. parsed-literal::

      39	 1.0122973e+00	 1.5820231e-01	 1.0595186e+00	 1.7249011e-01	[ 1.0118828e+00]	 2.0309758e-01
      40	 1.0235955e+00	 1.5741721e-01	 1.0712584e+00	 1.7074271e-01	[ 1.0216335e+00]	 2.0267010e-01


.. parsed-literal::

      41	 1.0353620e+00	 1.5661757e-01	 1.0834533e+00	 1.6915144e-01	[ 1.0319236e+00]	 1.8135571e-01
      42	 1.0603724e+00	 1.5386966e-01	 1.1097902e+00	 1.6553733e-01	[ 1.0538177e+00]	 1.9786549e-01


.. parsed-literal::

      43	 1.0680868e+00	 1.5289808e-01	 1.1178568e+00	 1.6498199e-01	[ 1.0606707e+00]	 3.1451583e-01
      44	 1.0779749e+00	 1.5192472e-01	 1.1274650e+00	 1.6491834e-01	[ 1.0696115e+00]	 1.9070816e-01


.. parsed-literal::

      45	 1.0908354e+00	 1.5128128e-01	 1.1406716e+00	 1.6548845e-01	[ 1.0793898e+00]	 2.0967650e-01


.. parsed-literal::

      46	 1.0992681e+00	 1.5119579e-01	 1.1493459e+00	 1.6621196e-01	[ 1.0848329e+00]	 2.1343756e-01


.. parsed-literal::

      47	 1.1111767e+00	 1.5123478e-01	 1.1617275e+00	 1.6699722e-01	[ 1.0957603e+00]	 2.0845652e-01


.. parsed-literal::

      48	 1.1223028e+00	 1.5069316e-01	 1.1730986e+00	 1.6674426e-01	[ 1.1043767e+00]	 2.1898246e-01


.. parsed-literal::

      49	 1.1319230e+00	 1.5014682e-01	 1.1826932e+00	 1.6603986e-01	[ 1.1148753e+00]	 2.0859361e-01


.. parsed-literal::

      50	 1.1404270e+00	 1.5010090e-01	 1.1911529e+00	 1.6591853e-01	[ 1.1204017e+00]	 2.1547937e-01


.. parsed-literal::

      51	 1.1479964e+00	 1.4968068e-01	 1.1985798e+00	 1.6545334e-01	[ 1.1287985e+00]	 2.0446277e-01


.. parsed-literal::

      52	 1.1557209e+00	 1.4927404e-01	 1.2062139e+00	 1.6522603e-01	[ 1.1350028e+00]	 2.1200514e-01


.. parsed-literal::

      53	 1.1654014e+00	 1.4923028e-01	 1.2163006e+00	 1.6536046e-01	[ 1.1410744e+00]	 2.1037769e-01


.. parsed-literal::

      54	 1.1751268e+00	 1.4774639e-01	 1.2268744e+00	 1.6422143e-01	[ 1.1428802e+00]	 2.1161270e-01


.. parsed-literal::

      55	 1.1855620e+00	 1.4692544e-01	 1.2372025e+00	 1.6364211e-01	[ 1.1546339e+00]	 2.0737123e-01


.. parsed-literal::

      56	 1.1934160e+00	 1.4587898e-01	 1.2452387e+00	 1.6290701e-01	[ 1.1647293e+00]	 2.0919275e-01


.. parsed-literal::

      57	 1.2028633e+00	 1.4440878e-01	 1.2548769e+00	 1.6170914e-01	[ 1.1754887e+00]	 2.0731068e-01


.. parsed-literal::

      58	 1.2104886e+00	 1.4331832e-01	 1.2630303e+00	 1.6151961e-01	[ 1.1789544e+00]	 2.1200752e-01
      59	 1.2200003e+00	 1.4274186e-01	 1.2721912e+00	 1.6101278e-01	[ 1.1874033e+00]	 1.7942739e-01


.. parsed-literal::

      60	 1.2275529e+00	 1.4237845e-01	 1.2797188e+00	 1.6077114e-01	[ 1.1922317e+00]	 2.0630765e-01


.. parsed-literal::

      61	 1.2343117e+00	 1.4235694e-01	 1.2864849e+00	 1.6078048e-01	[ 1.1952872e+00]	 2.0220017e-01


.. parsed-literal::

      62	 1.2412875e+00	 1.4181126e-01	 1.2935247e+00	 1.6008490e-01	[ 1.1999134e+00]	 2.0887852e-01


.. parsed-literal::

      63	 1.2473422e+00	 1.4110428e-01	 1.2997563e+00	 1.5914240e-01	[ 1.2032623e+00]	 2.0547748e-01
      64	 1.2547121e+00	 1.4006793e-01	 1.3074792e+00	 1.5776117e-01	[ 1.2073741e+00]	 1.9815373e-01


.. parsed-literal::

      65	 1.2630449e+00	 1.3882746e-01	 1.3162901e+00	 1.5582284e-01	[ 1.2138087e+00]	 2.0403290e-01


.. parsed-literal::

      66	 1.2709793e+00	 1.3808118e-01	 1.3245820e+00	 1.5456802e-01	[ 1.2216906e+00]	 2.0831800e-01
      67	 1.2776363e+00	 1.3756650e-01	 1.3313539e+00	 1.5407155e-01	[ 1.2274088e+00]	 1.8899512e-01


.. parsed-literal::

      68	 1.2835196e+00	 1.3748234e-01	 1.3373369e+00	 1.5415993e-01	[ 1.2303880e+00]	 1.9632816e-01
      69	 1.2897633e+00	 1.3694976e-01	 1.3438574e+00	 1.5352380e-01	[ 1.2328508e+00]	 1.9894195e-01


.. parsed-literal::

      70	 1.2977127e+00	 1.3618087e-01	 1.3521503e+00	 1.5272334e-01	[ 1.2354098e+00]	 2.0959187e-01
      71	 1.3046655e+00	 1.3480613e-01	 1.3593870e+00	 1.5128507e-01	  1.2340642e+00 	 2.0110488e-01


.. parsed-literal::

      72	 1.3105924e+00	 1.3435840e-01	 1.3651862e+00	 1.5078440e-01	[ 1.2431883e+00]	 2.0940518e-01
      73	 1.3144092e+00	 1.3396390e-01	 1.3690175e+00	 1.5035321e-01	[ 1.2489534e+00]	 1.7002916e-01


.. parsed-literal::

      74	 1.3183098e+00	 1.3339650e-01	 1.3729410e+00	 1.4977409e-01	[ 1.2541612e+00]	 1.9910836e-01
      75	 1.3220326e+00	 1.3296807e-01	 1.3766760e+00	 1.4953004e-01	[ 1.2562605e+00]	 1.9417715e-01


.. parsed-literal::

      76	 1.3251256e+00	 1.3269468e-01	 1.3798018e+00	 1.4929775e-01	[ 1.2571988e+00]	 2.1336460e-01


.. parsed-literal::

      77	 1.3328367e+00	 1.3208251e-01	 1.3875508e+00	 1.4874035e-01	[ 1.2586169e+00]	 2.0739269e-01


.. parsed-literal::

      78	 1.3394272e+00	 1.3159978e-01	 1.3942228e+00	 1.4877083e-01	[ 1.2607442e+00]	 2.0925140e-01
      79	 1.3457679e+00	 1.3114454e-01	 1.4006221e+00	 1.4876003e-01	[ 1.2622671e+00]	 1.9474125e-01


.. parsed-literal::

      80	 1.3503260e+00	 1.3086084e-01	 1.4052240e+00	 1.4871238e-01	[ 1.2654616e+00]	 2.1174264e-01


.. parsed-literal::

      81	 1.3547873e+00	 1.3050142e-01	 1.4098860e+00	 1.4863410e-01	[ 1.2676068e+00]	 2.0330143e-01


.. parsed-literal::

      82	 1.3596579e+00	 1.2989886e-01	 1.4151263e+00	 1.4830853e-01	[ 1.2701178e+00]	 2.1342254e-01


.. parsed-literal::

      83	 1.3644794e+00	 1.2921302e-01	 1.4201175e+00	 1.4759808e-01	[ 1.2707325e+00]	 2.1552801e-01


.. parsed-literal::

      84	 1.3685480e+00	 1.2903464e-01	 1.4241570e+00	 1.4734771e-01	[ 1.2741335e+00]	 2.1412849e-01


.. parsed-literal::

      85	 1.3730781e+00	 1.2836218e-01	 1.4289376e+00	 1.4672188e-01	  1.2725661e+00 	 2.1787190e-01
      86	 1.3775730e+00	 1.2816420e-01	 1.4334807e+00	 1.4647217e-01	[ 1.2744710e+00]	 1.8097019e-01


.. parsed-literal::

      87	 1.3814349e+00	 1.2785236e-01	 1.4374167e+00	 1.4630092e-01	[ 1.2757409e+00]	 2.0935845e-01


.. parsed-literal::

      88	 1.3853954e+00	 1.2739112e-01	 1.4414694e+00	 1.4582899e-01	[ 1.2760271e+00]	 2.1746397e-01


.. parsed-literal::

      89	 1.3890465e+00	 1.2698961e-01	 1.4451593e+00	 1.4565422e-01	  1.2731822e+00 	 2.1147561e-01
      90	 1.3921619e+00	 1.2677244e-01	 1.4481708e+00	 1.4536768e-01	[ 1.2775689e+00]	 1.9962549e-01


.. parsed-literal::

      91	 1.3975559e+00	 1.2612223e-01	 1.4536271e+00	 1.4445420e-01	[ 1.2810821e+00]	 2.0594716e-01


.. parsed-literal::

      92	 1.4010186e+00	 1.2563638e-01	 1.4571059e+00	 1.4374994e-01	[ 1.2844628e+00]	 2.1135116e-01
      93	 1.4053386e+00	 1.2525901e-01	 1.4615767e+00	 1.4316058e-01	  1.2823277e+00 	 2.0072126e-01


.. parsed-literal::

      94	 1.4091992e+00	 1.2498461e-01	 1.4655277e+00	 1.4274088e-01	  1.2809236e+00 	 2.0893312e-01


.. parsed-literal::

      95	 1.4127967e+00	 1.2472480e-01	 1.4693304e+00	 1.4222620e-01	  1.2828544e+00 	 2.1473622e-01


.. parsed-literal::

      96	 1.4158616e+00	 1.2451309e-01	 1.4724337e+00	 1.4179137e-01	[ 1.2848402e+00]	 2.0743012e-01
      97	 1.4187377e+00	 1.2435533e-01	 1.4752933e+00	 1.4150964e-01	[ 1.2878941e+00]	 1.8622065e-01


.. parsed-literal::

      98	 1.4231241e+00	 1.2396209e-01	 1.4796570e+00	 1.4088488e-01	[ 1.2948531e+00]	 1.9896936e-01
      99	 1.4262911e+00	 1.2377348e-01	 1.4829032e+00	 1.4089220e-01	[ 1.2976325e+00]	 1.8908620e-01


.. parsed-literal::

     100	 1.4298136e+00	 1.2350731e-01	 1.4863563e+00	 1.4068165e-01	[ 1.3023856e+00]	 2.0990515e-01


.. parsed-literal::

     101	 1.4325074e+00	 1.2331289e-01	 1.4890914e+00	 1.4067511e-01	[ 1.3038703e+00]	 2.0426941e-01


.. parsed-literal::

     102	 1.4354003e+00	 1.2318181e-01	 1.4920482e+00	 1.4079523e-01	[ 1.3046339e+00]	 2.0550108e-01
     103	 1.4376431e+00	 1.2323943e-01	 1.4943638e+00	 1.4128276e-01	  1.3018596e+00 	 1.7770123e-01


.. parsed-literal::

     104	 1.4421907e+00	 1.2305841e-01	 1.4988148e+00	 1.4102789e-01	[ 1.3051462e+00]	 2.0603251e-01


.. parsed-literal::

     105	 1.4441987e+00	 1.2296330e-01	 1.5007385e+00	 1.4076641e-01	[ 1.3065086e+00]	 2.1275830e-01
     106	 1.4465270e+00	 1.2279088e-01	 1.5030049e+00	 1.4043465e-01	  1.3060761e+00 	 1.9476938e-01


.. parsed-literal::

     107	 1.4486028e+00	 1.2267432e-01	 1.5052436e+00	 1.4034900e-01	  1.3036720e+00 	 2.0328355e-01


.. parsed-literal::

     108	 1.4517934e+00	 1.2244672e-01	 1.5083642e+00	 1.4000022e-01	  1.3020027e+00 	 2.1800685e-01


.. parsed-literal::

     109	 1.4538595e+00	 1.2230687e-01	 1.5104816e+00	 1.3992587e-01	  1.3012471e+00 	 2.1340132e-01


.. parsed-literal::

     110	 1.4559874e+00	 1.2217410e-01	 1.5127217e+00	 1.3994128e-01	  1.3006759e+00 	 2.1090555e-01


.. parsed-literal::

     111	 1.4594363e+00	 1.2200058e-01	 1.5162654e+00	 1.4003274e-01	  1.3009132e+00 	 2.1601295e-01


.. parsed-literal::

     112	 1.4626373e+00	 1.2181916e-01	 1.5195428e+00	 1.4006504e-01	  1.3005748e+00 	 3.3034372e-01


.. parsed-literal::

     113	 1.4667529e+00	 1.2166063e-01	 1.5237587e+00	 1.4023873e-01	  1.3030530e+00 	 2.1067739e-01


.. parsed-literal::

     114	 1.4689615e+00	 1.2153641e-01	 1.5259409e+00	 1.4002366e-01	  1.3064366e+00 	 2.1040368e-01


.. parsed-literal::

     115	 1.4719736e+00	 1.2131984e-01	 1.5289857e+00	 1.3957461e-01	[ 1.3082788e+00]	 2.0996284e-01
     116	 1.4739774e+00	 1.2116846e-01	 1.5310954e+00	 1.3916815e-01	[ 1.3112392e+00]	 2.0693111e-01


.. parsed-literal::

     117	 1.4761493e+00	 1.2102672e-01	 1.5333242e+00	 1.3904732e-01	  1.3101496e+00 	 1.9536376e-01


.. parsed-literal::

     118	 1.4789675e+00	 1.2075580e-01	 1.5363331e+00	 1.3891290e-01	  1.3070950e+00 	 2.0970869e-01
     119	 1.4812406e+00	 1.2054438e-01	 1.5386921e+00	 1.3881520e-01	  1.3049922e+00 	 1.8841195e-01


.. parsed-literal::

     120	 1.4837496e+00	 1.2038480e-01	 1.5414217e+00	 1.3903073e-01	  1.2980877e+00 	 2.1158814e-01


.. parsed-literal::

     121	 1.4858910e+00	 1.2023637e-01	 1.5434730e+00	 1.3886354e-01	  1.3008333e+00 	 2.0894027e-01


.. parsed-literal::

     122	 1.4867942e+00	 1.2024729e-01	 1.5442523e+00	 1.3885745e-01	  1.3038842e+00 	 2.0626354e-01
     123	 1.4889716e+00	 1.2016002e-01	 1.5463714e+00	 1.3882994e-01	  1.3052669e+00 	 1.8727350e-01


.. parsed-literal::

     124	 1.4911204e+00	 1.2005417e-01	 1.5485913e+00	 1.3887477e-01	  1.3040451e+00 	 1.9449830e-01
     125	 1.4928820e+00	 1.1993938e-01	 1.5503928e+00	 1.3875470e-01	  1.3014815e+00 	 1.9870830e-01


.. parsed-literal::

     126	 1.4941616e+00	 1.1988012e-01	 1.5517348e+00	 1.3866955e-01	  1.2985616e+00 	 1.8710947e-01
     127	 1.4960586e+00	 1.1981784e-01	 1.5536993e+00	 1.3859749e-01	  1.2946198e+00 	 1.8254447e-01


.. parsed-literal::

     128	 1.4972609e+00	 1.1981427e-01	 1.5549483e+00	 1.3861578e-01	  1.2909053e+00 	 3.3128095e-01
     129	 1.4989074e+00	 1.1978159e-01	 1.5565539e+00	 1.3862519e-01	  1.2915442e+00 	 1.9409823e-01


.. parsed-literal::

     130	 1.5003812e+00	 1.1973818e-01	 1.5579951e+00	 1.3869878e-01	  1.2920449e+00 	 2.0859742e-01


.. parsed-literal::

     131	 1.5020073e+00	 1.1969946e-01	 1.5595490e+00	 1.3889236e-01	  1.2932639e+00 	 2.1721840e-01


.. parsed-literal::

     132	 1.5038150e+00	 1.1963423e-01	 1.5613543e+00	 1.3903533e-01	  1.2942624e+00 	 2.0887899e-01
     133	 1.5054819e+00	 1.1951444e-01	 1.5630217e+00	 1.3906071e-01	  1.2949310e+00 	 1.9477987e-01


.. parsed-literal::

     134	 1.5070166e+00	 1.1946374e-01	 1.5646214e+00	 1.3917142e-01	  1.2931832e+00 	 2.0874882e-01
     135	 1.5084076e+00	 1.1934653e-01	 1.5659962e+00	 1.3904895e-01	  1.2939794e+00 	 1.8406749e-01


.. parsed-literal::

     136	 1.5099564e+00	 1.1926418e-01	 1.5675454e+00	 1.3896638e-01	  1.2932356e+00 	 2.1105266e-01
     137	 1.5118697e+00	 1.1914880e-01	 1.5695024e+00	 1.3888380e-01	  1.2918875e+00 	 1.8240356e-01


.. parsed-literal::

     138	 1.5136455e+00	 1.1908322e-01	 1.5712815e+00	 1.3893604e-01	  1.2895407e+00 	 2.0196366e-01
     139	 1.5149506e+00	 1.1901683e-01	 1.5725489e+00	 1.3892025e-01	  1.2907890e+00 	 1.8611646e-01


.. parsed-literal::

     140	 1.5170082e+00	 1.1885289e-01	 1.5745954e+00	 1.3886055e-01	  1.2909826e+00 	 2.1190882e-01
     141	 1.5183489e+00	 1.1876158e-01	 1.5759687e+00	 1.3882288e-01	  1.2916326e+00 	 1.8605828e-01


.. parsed-literal::

     142	 1.5199765e+00	 1.1868347e-01	 1.5776303e+00	 1.3872595e-01	  1.2903264e+00 	 1.9710302e-01
     143	 1.5214144e+00	 1.1861226e-01	 1.5791090e+00	 1.3857422e-01	  1.2886916e+00 	 1.9384360e-01


.. parsed-literal::

     144	 1.5228111e+00	 1.1859838e-01	 1.5805711e+00	 1.3845769e-01	  1.2865871e+00 	 1.9048667e-01
     145	 1.5246394e+00	 1.1856223e-01	 1.5824876e+00	 1.3833347e-01	  1.2846033e+00 	 1.7684102e-01


.. parsed-literal::

     146	 1.5263422e+00	 1.1852777e-01	 1.5842497e+00	 1.3820906e-01	  1.2840779e+00 	 1.9962525e-01


.. parsed-literal::

     147	 1.5276537e+00	 1.1856916e-01	 1.5855849e+00	 1.3823330e-01	  1.2850372e+00 	 2.0282984e-01


.. parsed-literal::

     148	 1.5286743e+00	 1.1850788e-01	 1.5865910e+00	 1.3817075e-01	  1.2878765e+00 	 2.0746613e-01


.. parsed-literal::

     149	 1.5308600e+00	 1.1844571e-01	 1.5888029e+00	 1.3804140e-01	  1.2944111e+00 	 2.1741104e-01
     150	 1.5316075e+00	 1.1837499e-01	 1.5896326e+00	 1.3782421e-01	  1.3008937e+00 	 1.9585609e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.08 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4118143310>



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

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 950 ms, sys: 42 ms, total: 992 ms
    Wall time: 372 ms


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




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

