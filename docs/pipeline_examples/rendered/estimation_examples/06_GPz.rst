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
       1	-3.4426588e-01	 3.2035603e-01	-3.3395052e-01	 3.2143841e-01	[-3.3631517e-01]	 4.5643330e-01


.. parsed-literal::

       2	-2.7343928e-01	 3.1016221e-01	-2.4963835e-01	 3.1142869e-01	[-2.5389509e-01]	 2.2744513e-01


.. parsed-literal::

       3	-2.2840514e-01	 2.8931828e-01	-1.8658013e-01	 2.9009933e-01	[-1.9050137e-01]	 2.7189112e-01
       4	-1.9047191e-01	 2.6514413e-01	-1.4954978e-01	 2.6464331e-01	[-1.4957239e-01]	 1.8416381e-01


.. parsed-literal::

       5	-1.0016156e-01	 2.5623260e-01	-6.6401309e-02	 2.5909192e-01	[-8.0088520e-02]	 2.0379376e-01


.. parsed-literal::

       6	-6.9201984e-02	 2.5132373e-01	-3.8734352e-02	 2.5353569e-01	[-4.7082752e-02]	 2.0937991e-01


.. parsed-literal::

       7	-5.1683860e-02	 2.4862668e-01	-2.7470061e-02	 2.5042688e-01	[-3.4968953e-02]	 2.0300364e-01


.. parsed-literal::

       8	-3.8681333e-02	 2.4641330e-01	-1.8216936e-02	 2.4766145e-01	[-2.3652121e-02]	 2.0458913e-01
       9	-2.4850485e-02	 2.4379336e-01	-7.0255499e-03	 2.4487776e-01	[-1.1811652e-02]	 2.0147324e-01


.. parsed-literal::

      10	-1.4916221e-02	 2.4201764e-01	 9.3559584e-04	 2.4419845e-01	[-8.8652303e-03]	 2.0335031e-01


.. parsed-literal::

      11	-1.0264336e-02	 2.4119085e-01	 4.8565018e-03	 2.4462836e-01	 -8.9827809e-03 	 2.0519161e-01


.. parsed-literal::

      12	-7.8530270e-03	 2.4067149e-01	 7.1362915e-03	 2.4372499e-01	[-5.6546992e-03]	 2.0437407e-01
      13	-3.2967530e-03	 2.3976582e-01	 1.1604351e-02	 2.4201833e-01	[ 2.0587265e-03]	 2.0442581e-01


.. parsed-literal::

      14	 3.0272160e-03	 2.3840414e-01	 1.8924166e-02	 2.4065642e-01	[ 8.9564913e-03]	 1.9998980e-01


.. parsed-literal::

      15	 2.1359425e-01	 2.2283847e-01	 2.4417816e-01	 2.2174930e-01	[ 2.4153332e-01]	 3.1697178e-01


.. parsed-literal::

      16	 2.5141184e-01	 2.1698713e-01	 2.8293886e-01	 2.1925863e-01	[ 2.6969468e-01]	 3.1697464e-01


.. parsed-literal::

      17	 2.9497100e-01	 2.1076923e-01	 3.2631115e-01	 2.1323401e-01	[ 3.1329783e-01]	 2.1174383e-01


.. parsed-literal::

      18	 3.5261494e-01	 2.0694039e-01	 3.8514394e-01	 2.1065319e-01	[ 3.7069371e-01]	 2.0553637e-01


.. parsed-literal::

      19	 4.1273733e-01	 2.0322987e-01	 4.4835614e-01	 2.0634635e-01	[ 4.3111401e-01]	 2.0542812e-01


.. parsed-literal::

      20	 4.6143135e-01	 2.0500601e-01	 4.9819233e-01	 2.0853667e-01	[ 4.7545135e-01]	 2.1814299e-01


.. parsed-literal::

      21	 5.2410358e-01	 2.0333237e-01	 5.6206611e-01	 2.0725456e-01	[ 5.3734071e-01]	 2.0507455e-01
      22	 5.8570154e-01	 2.0113775e-01	 6.2632021e-01	 2.0688046e-01	[ 5.8927097e-01]	 2.0237112e-01


.. parsed-literal::

      23	 6.1552081e-01	 1.9821130e-01	 6.5747321e-01	 2.0566005e-01	[ 6.2491382e-01]	 2.0413017e-01


.. parsed-literal::

      24	 6.4420820e-01	 1.9470189e-01	 6.8484358e-01	 2.0003155e-01	[ 6.5117644e-01]	 2.0614100e-01


.. parsed-literal::

      25	 6.9045096e-01	 1.9380791e-01	 7.3039757e-01	 1.9562362e-01	[ 6.9563306e-01]	 2.0569181e-01


.. parsed-literal::

      26	 7.2673007e-01	 1.9633641e-01	 7.6576493e-01	 1.9898531e-01	[ 7.2571512e-01]	 2.0622349e-01


.. parsed-literal::

      27	 7.6099587e-01	 1.9873143e-01	 8.0154319e-01	 2.0129527e-01	[ 7.5569733e-01]	 2.1467972e-01
      28	 7.8439758e-01	 2.0819658e-01	 8.2703111e-01	 2.0961513e-01	[ 7.8947980e-01]	 1.8320036e-01


.. parsed-literal::

      29	 8.0729102e-01	 2.0285873e-01	 8.5388869e-01	 2.0222226e-01	[ 8.1035352e-01]	 2.0096660e-01
      30	 8.5628236e-01	 1.9427114e-01	 8.9964316e-01	 1.9257887e-01	[ 8.5185695e-01]	 2.0256853e-01


.. parsed-literal::

      31	 8.7817208e-01	 1.9079912e-01	 9.2159909e-01	 1.8899843e-01	[ 8.7331247e-01]	 2.0478249e-01
      32	 9.1788773e-01	 1.8821415e-01	 9.6254724e-01	 1.8643421e-01	[ 9.0692431e-01]	 1.9139266e-01


.. parsed-literal::

      33	 9.3452510e-01	 1.8394503e-01	 9.8182039e-01	 1.8088605e-01	[ 9.2847627e-01]	 2.0840645e-01


.. parsed-literal::

      34	 9.5970998e-01	 1.8219414e-01	 1.0065331e+00	 1.8065928e-01	[ 9.4651620e-01]	 2.1718478e-01


.. parsed-literal::

      35	 9.7308247e-01	 1.7905047e-01	 1.0195337e+00	 1.7819067e-01	[ 9.6070696e-01]	 2.1940708e-01


.. parsed-literal::

      36	 9.8994149e-01	 1.7511804e-01	 1.0367174e+00	 1.7513304e-01	[ 9.7980684e-01]	 2.0718884e-01
      37	 1.0044415e+00	 1.7154301e-01	 1.0524588e+00	 1.7217228e-01	[ 9.8540891e-01]	 1.9278264e-01


.. parsed-literal::

      38	 1.0163188e+00	 1.7104759e-01	 1.0647977e+00	 1.7162213e-01	[ 9.9260029e-01]	 2.0464611e-01


.. parsed-literal::

      39	 1.0241755e+00	 1.7074505e-01	 1.0732095e+00	 1.7136545e-01	[ 9.9775349e-01]	 2.1451831e-01


.. parsed-literal::

      40	 1.0397773e+00	 1.6965680e-01	 1.0893796e+00	 1.7024194e-01	[ 1.0109224e+00]	 2.1267152e-01


.. parsed-literal::

      41	 1.0602530e+00	 1.6678750e-01	 1.1105887e+00	 1.6748609e-01	[ 1.0345112e+00]	 2.0190787e-01


.. parsed-literal::

      42	 1.0721145e+00	 1.6623031e-01	 1.1219635e+00	 1.6662298e-01	[ 1.0530409e+00]	 2.0661426e-01


.. parsed-literal::

      43	 1.0821076e+00	 1.6556931e-01	 1.1315243e+00	 1.6613194e-01	[ 1.0639151e+00]	 2.2723079e-01
      44	 1.0967164e+00	 1.6416345e-01	 1.1461721e+00	 1.6480190e-01	[ 1.0811427e+00]	 1.9719243e-01


.. parsed-literal::

      45	 1.1100565e+00	 1.6278050e-01	 1.1597241e+00	 1.6320761e-01	[ 1.0978767e+00]	 2.0056748e-01


.. parsed-literal::

      46	 1.1271900e+00	 1.6040010e-01	 1.1773594e+00	 1.6036968e-01	[ 1.1125612e+00]	 2.1717119e-01


.. parsed-literal::

      47	 1.1320468e+00	 1.5863807e-01	 1.1822283e+00	 1.5856908e-01	[ 1.1209956e+00]	 2.1507335e-01
      48	 1.1429845e+00	 1.5766521e-01	 1.1930187e+00	 1.5761293e-01	[ 1.1273115e+00]	 1.8871236e-01


.. parsed-literal::

      49	 1.1499082e+00	 1.5682249e-01	 1.2002839e+00	 1.5667701e-01	[ 1.1317330e+00]	 2.1112847e-01
      50	 1.1608890e+00	 1.5499399e-01	 1.2119389e+00	 1.5451637e-01	[ 1.1397812e+00]	 1.8099546e-01


.. parsed-literal::

      51	 1.1771214e+00	 1.5237073e-01	 1.2285783e+00	 1.5096773e-01	[ 1.1606723e+00]	 2.1390748e-01


.. parsed-literal::

      52	 1.1892929e+00	 1.5039923e-01	 1.2426215e+00	 1.4831451e-01	[ 1.1681624e+00]	 2.1263552e-01
      53	 1.2007791e+00	 1.5019943e-01	 1.2534540e+00	 1.4748873e-01	[ 1.1831141e+00]	 1.9337177e-01


.. parsed-literal::

      54	 1.2073788e+00	 1.4955911e-01	 1.2598503e+00	 1.4658724e-01	[ 1.1916651e+00]	 2.0315361e-01


.. parsed-literal::

      55	 1.2201653e+00	 1.4888432e-01	 1.2730973e+00	 1.4533773e-01	[ 1.2036504e+00]	 2.1905255e-01


.. parsed-literal::

      56	 1.2285699e+00	 1.4853065e-01	 1.2815404e+00	 1.4411429e-01	[ 1.2117748e+00]	 3.3775854e-01


.. parsed-literal::

      57	 1.2399271e+00	 1.4797135e-01	 1.2929950e+00	 1.4360934e-01	[ 1.2178891e+00]	 2.1273732e-01


.. parsed-literal::

      58	 1.2493955e+00	 1.4703650e-01	 1.3037116e+00	 1.4240530e-01	  1.2163595e+00 	 2.0954204e-01
      59	 1.2568695e+00	 1.4664957e-01	 1.3110054e+00	 1.4204117e-01	[ 1.2250406e+00]	 1.9151926e-01


.. parsed-literal::

      60	 1.2642500e+00	 1.4583704e-01	 1.3183363e+00	 1.4124224e-01	[ 1.2319403e+00]	 2.1495962e-01


.. parsed-literal::

      61	 1.2767692e+00	 1.4497187e-01	 1.3310101e+00	 1.4028353e-01	[ 1.2441862e+00]	 2.1217251e-01


.. parsed-literal::

      62	 1.2846592e+00	 1.4323368e-01	 1.3388813e+00	 1.3868765e-01	  1.2421789e+00 	 2.1424484e-01


.. parsed-literal::

      63	 1.2912387e+00	 1.4324913e-01	 1.3452859e+00	 1.3901172e-01	[ 1.2506966e+00]	 2.0637655e-01
      64	 1.3002405e+00	 1.4319647e-01	 1.3546849e+00	 1.3909659e-01	[ 1.2510960e+00]	 1.9863939e-01


.. parsed-literal::

      65	 1.3072229e+00	 1.4256687e-01	 1.3619018e+00	 1.3900415e-01	[ 1.2533315e+00]	 2.0875406e-01


.. parsed-literal::

      66	 1.3152969e+00	 1.4139048e-01	 1.3704489e+00	 1.3890043e-01	  1.2505923e+00 	 2.1010685e-01
      67	 1.3246733e+00	 1.4032573e-01	 1.3798375e+00	 1.3854892e-01	[ 1.2579525e+00]	 2.0199966e-01


.. parsed-literal::

      68	 1.3330996e+00	 1.3950933e-01	 1.3883675e+00	 1.3843093e-01	[ 1.2655934e+00]	 1.9235277e-01


.. parsed-literal::

      69	 1.3403827e+00	 1.3926969e-01	 1.3957656e+00	 1.3840900e-01	[ 1.2657358e+00]	 2.0810461e-01
      70	 1.3476524e+00	 1.3907965e-01	 1.4031482e+00	 1.3858620e-01	[ 1.2740310e+00]	 1.9142175e-01


.. parsed-literal::

      71	 1.3530123e+00	 1.3897082e-01	 1.4086609e+00	 1.3851753e-01	[ 1.2770900e+00]	 2.0233965e-01


.. parsed-literal::

      72	 1.3602312e+00	 1.3849855e-01	 1.4161445e+00	 1.3836760e-01	[ 1.2783485e+00]	 2.0273161e-01


.. parsed-literal::

      73	 1.3666866e+00	 1.3818640e-01	 1.4228895e+00	 1.3833146e-01	  1.2750414e+00 	 2.0990038e-01
      74	 1.3741509e+00	 1.3761390e-01	 1.4302141e+00	 1.3801712e-01	[ 1.2801079e+00]	 1.9970775e-01


.. parsed-literal::

      75	 1.3776737e+00	 1.3734349e-01	 1.4336607e+00	 1.3785372e-01	[ 1.2831129e+00]	 1.8653655e-01


.. parsed-literal::

      76	 1.3831892e+00	 1.3699671e-01	 1.4393234e+00	 1.3725742e-01	[ 1.2846151e+00]	 2.0881295e-01


.. parsed-literal::

      77	 1.3898979e+00	 1.3613612e-01	 1.4461493e+00	 1.3578424e-01	[ 1.2848672e+00]	 2.0633841e-01
      78	 1.3955532e+00	 1.3622052e-01	 1.4518690e+00	 1.3557948e-01	[ 1.2890968e+00]	 2.0920658e-01


.. parsed-literal::

      79	 1.4000319e+00	 1.3610380e-01	 1.4564617e+00	 1.3532931e-01	[ 1.2897875e+00]	 1.8686938e-01


.. parsed-literal::

      80	 1.4063540e+00	 1.3579600e-01	 1.4630245e+00	 1.3461543e-01	  1.2884389e+00 	 2.0748901e-01
      81	 1.4100158e+00	 1.3550221e-01	 1.4667141e+00	 1.3362434e-01	[ 1.2903698e+00]	 1.9238067e-01


.. parsed-literal::

      82	 1.4152463e+00	 1.3508649e-01	 1.4718835e+00	 1.3314853e-01	[ 1.2938044e+00]	 2.1006274e-01
      83	 1.4184867e+00	 1.3472431e-01	 1.4751531e+00	 1.3259464e-01	[ 1.2965032e+00]	 1.9887185e-01


.. parsed-literal::

      84	 1.4213751e+00	 1.3435337e-01	 1.4781316e+00	 1.3211263e-01	[ 1.2970956e+00]	 2.0749736e-01
      85	 1.4261442e+00	 1.3367821e-01	 1.4830825e+00	 1.3113253e-01	[ 1.2996937e+00]	 2.0052695e-01


.. parsed-literal::

      86	 1.4313479e+00	 1.3304910e-01	 1.4882464e+00	 1.3032305e-01	[ 1.3031614e+00]	 2.0538139e-01
      87	 1.4339028e+00	 1.3299218e-01	 1.4907515e+00	 1.3025353e-01	[ 1.3046587e+00]	 1.7960620e-01


.. parsed-literal::

      88	 1.4385977e+00	 1.3274840e-01	 1.4955397e+00	 1.2983182e-01	[ 1.3065745e+00]	 1.8209147e-01


.. parsed-literal::

      89	 1.4402298e+00	 1.3274733e-01	 1.4973391e+00	 1.2968011e-01	  1.2973302e+00 	 2.0941424e-01
      90	 1.4444360e+00	 1.3247055e-01	 1.5013973e+00	 1.2937631e-01	[ 1.3068137e+00]	 1.7494655e-01


.. parsed-literal::

      91	 1.4469344e+00	 1.3218455e-01	 1.5038982e+00	 1.2900550e-01	[ 1.3105065e+00]	 2.0069456e-01
      92	 1.4499810e+00	 1.3185780e-01	 1.5070140e+00	 1.2859836e-01	[ 1.3127682e+00]	 1.9340920e-01


.. parsed-literal::

      93	 1.4546186e+00	 1.3135794e-01	 1.5118091e+00	 1.2804746e-01	  1.3124431e+00 	 2.0160127e-01


.. parsed-literal::

      94	 1.4577955e+00	 1.3106725e-01	 1.5151281e+00	 1.2767030e-01	[ 1.3147829e+00]	 2.9904246e-01


.. parsed-literal::

      95	 1.4614856e+00	 1.3073589e-01	 1.5188825e+00	 1.2750611e-01	  1.3117084e+00 	 2.2186565e-01
      96	 1.4648405e+00	 1.3037444e-01	 1.5222571e+00	 1.2733529e-01	  1.3095083e+00 	 1.9410682e-01


.. parsed-literal::

      97	 1.4677862e+00	 1.3004617e-01	 1.5251888e+00	 1.2728022e-01	  1.3079546e+00 	 2.1082830e-01
      98	 1.4706163e+00	 1.2945337e-01	 1.5280631e+00	 1.2694728e-01	  1.3039501e+00 	 1.9629884e-01


.. parsed-literal::

      99	 1.4738940e+00	 1.2924069e-01	 1.5312633e+00	 1.2677032e-01	  1.3023383e+00 	 2.0853543e-01


.. parsed-literal::

     100	 1.4763174e+00	 1.2917667e-01	 1.5336681e+00	 1.2667007e-01	  1.2999807e+00 	 2.0568466e-01


.. parsed-literal::

     101	 1.4791334e+00	 1.2897277e-01	 1.5365959e+00	 1.2650372e-01	  1.2915206e+00 	 2.1217608e-01


.. parsed-literal::

     102	 1.4811448e+00	 1.2909820e-01	 1.5388842e+00	 1.2636733e-01	  1.2687327e+00 	 2.1198034e-01


.. parsed-literal::

     103	 1.4839622e+00	 1.2888273e-01	 1.5416238e+00	 1.2619613e-01	  1.2755203e+00 	 2.1425295e-01


.. parsed-literal::

     104	 1.4857704e+00	 1.2877881e-01	 1.5434229e+00	 1.2604531e-01	  1.2769130e+00 	 2.0654893e-01
     105	 1.4885362e+00	 1.2859843e-01	 1.5462183e+00	 1.2573434e-01	  1.2740726e+00 	 1.9426370e-01


.. parsed-literal::

     106	 1.4920647e+00	 1.2814566e-01	 1.5499864e+00	 1.2496501e-01	  1.2610675e+00 	 1.9418383e-01


.. parsed-literal::

     107	 1.4950921e+00	 1.2785370e-01	 1.5531839e+00	 1.2466343e-01	  1.2442923e+00 	 2.0923185e-01
     108	 1.4970127e+00	 1.2772111e-01	 1.5550753e+00	 1.2457751e-01	  1.2464993e+00 	 1.9374728e-01


.. parsed-literal::

     109	 1.4991733e+00	 1.2746132e-01	 1.5572953e+00	 1.2448707e-01	  1.2390484e+00 	 2.0573592e-01
     110	 1.5013885e+00	 1.2719852e-01	 1.5596393e+00	 1.2424671e-01	  1.2369107e+00 	 1.9391537e-01


.. parsed-literal::

     111	 1.5042216e+00	 1.2693500e-01	 1.5625089e+00	 1.2411866e-01	  1.2322676e+00 	 2.1028090e-01


.. parsed-literal::

     112	 1.5067449e+00	 1.2680933e-01	 1.5650194e+00	 1.2416305e-01	  1.2288778e+00 	 2.0683980e-01


.. parsed-literal::

     113	 1.5090374e+00	 1.2667868e-01	 1.5672063e+00	 1.2400728e-01	  1.2339866e+00 	 2.1501017e-01
     114	 1.5105427e+00	 1.2666706e-01	 1.5686433e+00	 1.2393698e-01	  1.2351892e+00 	 2.0285892e-01


.. parsed-literal::

     115	 1.5133278e+00	 1.2638489e-01	 1.5715377e+00	 1.2364524e-01	  1.2273896e+00 	 2.0606923e-01
     116	 1.5141112e+00	 1.2647052e-01	 1.5725167e+00	 1.2353183e-01	  1.2075105e+00 	 1.9967699e-01


.. parsed-literal::

     117	 1.5162517e+00	 1.2626351e-01	 1.5745676e+00	 1.2344972e-01	  1.2144170e+00 	 1.9182372e-01
     118	 1.5172892e+00	 1.2609785e-01	 1.5756381e+00	 1.2334928e-01	  1.2122742e+00 	 2.0139217e-01


.. parsed-literal::

     119	 1.5190142e+00	 1.2582186e-01	 1.5774144e+00	 1.2311746e-01	  1.2058172e+00 	 1.8390727e-01


.. parsed-literal::

     120	 1.5214143e+00	 1.2550497e-01	 1.5798023e+00	 1.2281671e-01	  1.2002613e+00 	 2.0423126e-01


.. parsed-literal::

     121	 1.5230902e+00	 1.2521006e-01	 1.5815401e+00	 1.2245123e-01	  1.1828306e+00 	 3.1654239e-01
     122	 1.5253949e+00	 1.2503596e-01	 1.5837916e+00	 1.2221886e-01	  1.1823748e+00 	 1.8834448e-01


.. parsed-literal::

     123	 1.5270631e+00	 1.2500167e-01	 1.5854452e+00	 1.2207739e-01	  1.1827526e+00 	 2.0040560e-01
     124	 1.5293348e+00	 1.2501042e-01	 1.5878112e+00	 1.2188490e-01	  1.1755232e+00 	 2.0255446e-01


.. parsed-literal::

     125	 1.5297746e+00	 1.2497755e-01	 1.5885325e+00	 1.2154384e-01	  1.1783493e+00 	 2.0426559e-01
     126	 1.5323218e+00	 1.2499109e-01	 1.5909985e+00	 1.2155766e-01	  1.1707429e+00 	 2.0286632e-01


.. parsed-literal::

     127	 1.5333903e+00	 1.2496511e-01	 1.5920827e+00	 1.2150865e-01	  1.1661066e+00 	 2.0158792e-01


.. parsed-literal::

     128	 1.5351038e+00	 1.2493553e-01	 1.5938735e+00	 1.2138249e-01	  1.1593069e+00 	 2.1597743e-01


.. parsed-literal::

     129	 1.5372540e+00	 1.2489564e-01	 1.5960736e+00	 1.2132203e-01	  1.1499067e+00 	 2.1434236e-01
     130	 1.5389387e+00	 1.2478847e-01	 1.5978888e+00	 1.2115427e-01	  1.1402990e+00 	 1.9773555e-01


.. parsed-literal::

     131	 1.5407731e+00	 1.2466744e-01	 1.5996366e+00	 1.2114141e-01	  1.1439402e+00 	 2.1003652e-01
     132	 1.5419447e+00	 1.2458126e-01	 1.6007978e+00	 1.2114937e-01	  1.1433397e+00 	 1.8666148e-01


.. parsed-literal::

     133	 1.5434484e+00	 1.2431111e-01	 1.6023695e+00	 1.2109827e-01	  1.1384566e+00 	 1.9742084e-01
     134	 1.5450275e+00	 1.2408849e-01	 1.6041015e+00	 1.2121996e-01	  1.1228361e+00 	 1.9958210e-01


.. parsed-literal::

     135	 1.5465888e+00	 1.2396241e-01	 1.6056825e+00	 1.2115448e-01	  1.1189856e+00 	 2.0226359e-01


.. parsed-literal::

     136	 1.5477712e+00	 1.2387191e-01	 1.6069386e+00	 1.2115390e-01	  1.1129287e+00 	 2.1303582e-01
     137	 1.5489511e+00	 1.2381204e-01	 1.6081664e+00	 1.2119143e-01	  1.1078788e+00 	 1.9615269e-01


.. parsed-literal::

     138	 1.5494725e+00	 1.2357571e-01	 1.6089378e+00	 1.2140013e-01	  1.0988189e+00 	 2.1719432e-01
     139	 1.5516692e+00	 1.2365392e-01	 1.6110001e+00	 1.2134631e-01	  1.0976829e+00 	 1.8940830e-01


.. parsed-literal::

     140	 1.5521931e+00	 1.2366170e-01	 1.6114906e+00	 1.2132100e-01	  1.0970003e+00 	 2.0152903e-01
     141	 1.5534708e+00	 1.2366179e-01	 1.6127397e+00	 1.2122914e-01	  1.0920924e+00 	 2.0099568e-01


.. parsed-literal::

     142	 1.5552214e+00	 1.2366144e-01	 1.6144921e+00	 1.2104027e-01	  1.0799408e+00 	 2.0983267e-01


.. parsed-literal::

     143	 1.5560716e+00	 1.2367815e-01	 1.6155183e+00	 1.2076479e-01	  1.0602319e+00 	 2.0464373e-01


.. parsed-literal::

     144	 1.5578905e+00	 1.2362989e-01	 1.6172522e+00	 1.2071541e-01	  1.0581880e+00 	 2.2181964e-01


.. parsed-literal::

     145	 1.5586309e+00	 1.2359044e-01	 1.6180176e+00	 1.2066640e-01	  1.0545369e+00 	 2.0789790e-01


.. parsed-literal::

     146	 1.5598357e+00	 1.2354973e-01	 1.6192939e+00	 1.2053328e-01	  1.0412938e+00 	 2.0595407e-01


.. parsed-literal::

     147	 1.5605414e+00	 1.2338515e-01	 1.6202209e+00	 1.2020374e-01	  1.0198319e+00 	 2.0930076e-01


.. parsed-literal::

     148	 1.5624038e+00	 1.2343101e-01	 1.6220105e+00	 1.2015996e-01	  1.0079849e+00 	 2.0322657e-01


.. parsed-literal::

     149	 1.5632335e+00	 1.2342556e-01	 1.6228202e+00	 1.2010313e-01	  1.0023461e+00 	 2.1632814e-01


.. parsed-literal::

     150	 1.5641408e+00	 1.2336954e-01	 1.6237345e+00	 1.2000198e-01	  9.9401611e-01 	 2.1308517e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.12 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f626cfb7ca0>



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
    CPU times: user 974 ms, sys: 48.9 ms, total: 1.02 s
    Wall time: 382 ms


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

