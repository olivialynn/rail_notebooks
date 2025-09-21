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
       1	-3.5143566e-01	 3.2307880e-01	-3.4178897e-01	 3.1055432e-01	[-3.1881985e-01]	 4.5941401e-01


.. parsed-literal::

       2	-2.8251403e-01	 3.1335991e-01	-2.5946642e-01	 3.0058499e-01	[-2.2204846e-01]	 2.3050904e-01


.. parsed-literal::

       3	-2.3891427e-01	 2.9231069e-01	-1.9730424e-01	 2.7979014e-01	[-1.4457263e-01]	 2.8220010e-01
       4	-1.9303028e-01	 2.6907196e-01	-1.5035348e-01	 2.5466116e-01	[-7.4000301e-02]	 1.7279148e-01


.. parsed-literal::

       5	-1.0767342e-01	 2.5836579e-01	-7.7372353e-02	 2.4484862e-01	[-1.3897984e-02]	 2.0048428e-01


.. parsed-literal::

       6	-7.8312473e-02	 2.5426262e-01	-5.0555694e-02	 2.4048742e-01	[-2.0829828e-03]	 2.0251608e-01
       7	-6.0958695e-02	 2.5119592e-01	-3.7780548e-02	 2.3733200e-01	[ 1.3496930e-02]	 1.7596817e-01


.. parsed-literal::

       8	-4.8520813e-02	 2.4911674e-01	-2.8988155e-02	 2.3536671e-01	[ 2.4976601e-02]	 2.1435332e-01


.. parsed-literal::

       9	-3.6688409e-02	 2.4680358e-01	-1.9309914e-02	 2.3320561e-01	[ 3.2187141e-02]	 2.1219373e-01


.. parsed-literal::

      10	-2.8030217e-02	 2.4460849e-01	-1.2945560e-02	 2.3175101e-01	[ 4.4632518e-02]	 2.0836067e-01
      11	-2.0104424e-02	 2.4366532e-01	-5.4916024e-03	 2.3235166e-01	  4.0481399e-02 	 1.8902564e-01


.. parsed-literal::

      12	-1.7447820e-02	 2.4318035e-01	-3.1967753e-03	 2.3225943e-01	  4.0994855e-02 	 2.1003938e-01


.. parsed-literal::

      13	-1.2418198e-02	 2.4218139e-01	 1.7451808e-03	 2.3207398e-01	  4.2444658e-02 	 2.0850301e-01


.. parsed-literal::

      14	 7.7559019e-02	 2.2907859e-01	 9.6293132e-02	 2.2119694e-01	[ 1.2556750e-01]	 3.2544255e-01


.. parsed-literal::

      15	 1.0580915e-01	 2.2690783e-01	 1.2852704e-01	 2.1766143e-01	[ 1.4427034e-01]	 2.1346974e-01


.. parsed-literal::

      16	 1.9198682e-01	 2.2158473e-01	 2.1706459e-01	 2.1492403e-01	[ 2.2858853e-01]	 2.0974755e-01
      17	 2.4605531e-01	 2.1700523e-01	 2.7815601e-01	 2.0897597e-01	[ 3.0253934e-01]	 1.8711948e-01


.. parsed-literal::

      18	 3.3712799e-01	 2.1464095e-01	 3.6867827e-01	 2.0543922e-01	[ 3.8558675e-01]	 2.0210481e-01


.. parsed-literal::

      19	 3.8223801e-01	 2.1054025e-01	 4.1460154e-01	 2.0119227e-01	[ 4.3186887e-01]	 2.1529484e-01


.. parsed-literal::

      20	 4.6288113e-01	 2.0691594e-01	 4.9545485e-01	 1.9784918e-01	[ 5.0971163e-01]	 2.0779705e-01


.. parsed-literal::

      21	 5.5024723e-01	 2.0562117e-01	 5.8522365e-01	 1.9760937e-01	[ 5.8268155e-01]	 2.1475959e-01
      22	 6.2355896e-01	 2.0143768e-01	 6.6136163e-01	 1.9190085e-01	[ 6.4182633e-01]	 1.9949341e-01


.. parsed-literal::

      23	 6.5749196e-01	 1.9563150e-01	 6.9732820e-01	 1.8736377e-01	[ 6.6161379e-01]	 1.8309402e-01


.. parsed-literal::

      24	 6.9845350e-01	 1.9201010e-01	 7.3664312e-01	 1.8391932e-01	[ 6.9370434e-01]	 2.0809364e-01


.. parsed-literal::

      25	 7.1876338e-01	 1.9129937e-01	 7.5621790e-01	 1.8329704e-01	[ 7.1483846e-01]	 2.1938896e-01


.. parsed-literal::

      26	 7.3938522e-01	 1.9199809e-01	 7.7597708e-01	 1.8446285e-01	[ 7.4025498e-01]	 2.1177220e-01


.. parsed-literal::

      27	 7.6269574e-01	 1.9156816e-01	 8.0021253e-01	 1.8415914e-01	[ 7.5687000e-01]	 2.0227599e-01
      28	 7.8413427e-01	 1.9090368e-01	 8.2298710e-01	 1.8263476e-01	[ 7.7171420e-01]	 1.9941807e-01


.. parsed-literal::

      29	 8.0825226e-01	 1.9167014e-01	 8.4687889e-01	 1.8274831e-01	[ 8.0116925e-01]	 2.1120954e-01
      30	 8.4832585e-01	 1.8536226e-01	 8.8737583e-01	 1.7375552e-01	[ 8.5445565e-01]	 1.7482901e-01


.. parsed-literal::

      31	 8.7794214e-01	 1.8259196e-01	 9.1726792e-01	 1.6954416e-01	[ 8.8602984e-01]	 2.0757937e-01
      32	 8.9599014e-01	 1.8051343e-01	 9.3616188e-01	 1.6700693e-01	[ 9.0522524e-01]	 1.9965291e-01


.. parsed-literal::

      33	 9.1232100e-01	 1.7816650e-01	 9.5335518e-01	 1.6288277e-01	[ 9.2549854e-01]	 2.0637536e-01
      34	 9.3775776e-01	 1.7694810e-01	 9.8021273e-01	 1.6192771e-01	[ 9.4899110e-01]	 1.8732381e-01


.. parsed-literal::

      35	 9.4722842e-01	 1.8088809e-01	 9.9073136e-01	 1.6437802e-01	[ 9.5758184e-01]	 2.1707106e-01


.. parsed-literal::

      36	 9.6336821e-01	 1.7780205e-01	 1.0065296e+00	 1.6208046e-01	[ 9.7031402e-01]	 2.1273708e-01


.. parsed-literal::

      37	 9.7154407e-01	 1.7733230e-01	 1.0144989e+00	 1.6243191e-01	[ 9.7720615e-01]	 2.1940088e-01
      38	 9.8023872e-01	 1.7754744e-01	 1.0232874e+00	 1.6268104e-01	[ 9.8611434e-01]	 1.9352007e-01


.. parsed-literal::

      39	 1.0002860e+00	 1.7776543e-01	 1.0438807e+00	 1.6161758e-01	[ 1.0110920e+00]	 2.1491218e-01
      40	 1.0010530e+00	 1.8007432e-01	 1.0455694e+00	 1.6136588e-01	[ 1.0150752e+00]	 1.9642282e-01


.. parsed-literal::

      41	 1.0192407e+00	 1.7714543e-01	 1.0632756e+00	 1.5873095e-01	[ 1.0350356e+00]	 2.1612883e-01


.. parsed-literal::

      42	 1.0271642e+00	 1.7563859e-01	 1.0712950e+00	 1.5721429e-01	[ 1.0426371e+00]	 2.1666741e-01
      43	 1.0364048e+00	 1.7409809e-01	 1.0808033e+00	 1.5536131e-01	[ 1.0519641e+00]	 1.9077682e-01


.. parsed-literal::

      44	 1.0499430e+00	 1.7216438e-01	 1.0950209e+00	 1.5273957e-01	[ 1.0637453e+00]	 1.8393922e-01


.. parsed-literal::

      45	 1.0538871e+00	 1.7266835e-01	 1.1000152e+00	 1.5170260e-01	[ 1.0734562e+00]	 2.1544886e-01


.. parsed-literal::

      46	 1.0688429e+00	 1.7109709e-01	 1.1148754e+00	 1.5020671e-01	[ 1.0816731e+00]	 2.0586300e-01


.. parsed-literal::

      47	 1.0751296e+00	 1.7046953e-01	 1.1209091e+00	 1.4982624e-01	[ 1.0846220e+00]	 2.1080899e-01


.. parsed-literal::

      48	 1.0855491e+00	 1.6981908e-01	 1.1314910e+00	 1.4915281e-01	[ 1.0898266e+00]	 2.1946692e-01
      49	 1.0985582e+00	 1.6846559e-01	 1.1449801e+00	 1.4824736e-01	[ 1.1019662e+00]	 1.7782640e-01


.. parsed-literal::

      50	 1.1089021e+00	 1.6792119e-01	 1.1555971e+00	 1.4656541e-01	[ 1.1073646e+00]	 2.0966363e-01


.. parsed-literal::

      51	 1.1157486e+00	 1.6749995e-01	 1.1624603e+00	 1.4636220e-01	[ 1.1121254e+00]	 2.0650935e-01


.. parsed-literal::

      52	 1.1270868e+00	 1.6655242e-01	 1.1741079e+00	 1.4500781e-01	[ 1.1235565e+00]	 2.0737696e-01
      53	 1.1367553e+00	 1.6538413e-01	 1.1841735e+00	 1.4347607e-01	[ 1.1335094e+00]	 1.9793248e-01


.. parsed-literal::

      54	 1.1471993e+00	 1.6396978e-01	 1.1947309e+00	 1.4134139e-01	[ 1.1513983e+00]	 2.1492910e-01


.. parsed-literal::

      55	 1.1556397e+00	 1.6297978e-01	 1.2031193e+00	 1.4009461e-01	[ 1.1623209e+00]	 2.1357918e-01
      56	 1.1604009e+00	 1.6260360e-01	 1.2082419e+00	 1.3959469e-01	[ 1.1751020e+00]	 1.7796159e-01


.. parsed-literal::

      57	 1.1679691e+00	 1.6218147e-01	 1.2155894e+00	 1.3914919e-01	[ 1.1800445e+00]	 2.0727992e-01
      58	 1.1738993e+00	 1.6143594e-01	 1.2215988e+00	 1.3855183e-01	[ 1.1834907e+00]	 1.8983889e-01


.. parsed-literal::

      59	 1.1814728e+00	 1.6044266e-01	 1.2294843e+00	 1.3761121e-01	[ 1.1887066e+00]	 2.0752883e-01


.. parsed-literal::

      60	 1.1920327e+00	 1.5858726e-01	 1.2403715e+00	 1.3567868e-01	[ 1.1939166e+00]	 2.0561457e-01


.. parsed-literal::

      61	 1.2012586e+00	 1.5640718e-01	 1.2500276e+00	 1.3389531e-01	  1.1878650e+00 	 2.1072173e-01


.. parsed-literal::

      62	 1.2088827e+00	 1.5592371e-01	 1.2575708e+00	 1.3344307e-01	[ 1.1971781e+00]	 2.1148491e-01


.. parsed-literal::

      63	 1.2168884e+00	 1.5485073e-01	 1.2655858e+00	 1.3246809e-01	[ 1.2042836e+00]	 2.0925522e-01


.. parsed-literal::

      64	 1.2249973e+00	 1.5406118e-01	 1.2740842e+00	 1.3190840e-01	[ 1.2088909e+00]	 2.1468091e-01


.. parsed-literal::

      65	 1.2338990e+00	 1.5256070e-01	 1.2834663e+00	 1.3010321e-01	[ 1.2180635e+00]	 2.0823169e-01


.. parsed-literal::

      66	 1.2408255e+00	 1.5198625e-01	 1.2903409e+00	 1.2914260e-01	[ 1.2222958e+00]	 2.0799851e-01
      67	 1.2463005e+00	 1.5187804e-01	 1.2957642e+00	 1.2886652e-01	[ 1.2262489e+00]	 1.9807768e-01


.. parsed-literal::

      68	 1.2531073e+00	 1.5126612e-01	 1.3026076e+00	 1.2808900e-01	[ 1.2315061e+00]	 1.7537427e-01


.. parsed-literal::

      69	 1.2616710e+00	 1.5039557e-01	 1.3114357e+00	 1.2703636e-01	[ 1.2370610e+00]	 2.0808959e-01


.. parsed-literal::

      70	 1.2686881e+00	 1.4973955e-01	 1.3186755e+00	 1.2636451e-01	[ 1.2426079e+00]	 2.0626235e-01
      71	 1.2754000e+00	 1.4909193e-01	 1.3254841e+00	 1.2591457e-01	[ 1.2494253e+00]	 1.7716193e-01


.. parsed-literal::

      72	 1.2824253e+00	 1.4879893e-01	 1.3327477e+00	 1.2576682e-01	[ 1.2558290e+00]	 1.9037986e-01


.. parsed-literal::

      73	 1.2908581e+00	 1.4820920e-01	 1.3413800e+00	 1.2526903e-01	[ 1.2646707e+00]	 2.1549654e-01


.. parsed-literal::

      74	 1.2982388e+00	 1.4772737e-01	 1.3486630e+00	 1.2480588e-01	[ 1.2737464e+00]	 2.1406698e-01
      75	 1.3040674e+00	 1.4814454e-01	 1.3542366e+00	 1.2534180e-01	[ 1.2783518e+00]	 1.9490266e-01


.. parsed-literal::

      76	 1.3086609e+00	 1.4816868e-01	 1.3586264e+00	 1.2522669e-01	[ 1.2787425e+00]	 2.1572399e-01


.. parsed-literal::

      77	 1.3129841e+00	 1.4815318e-01	 1.3630371e+00	 1.2505775e-01	[ 1.2821659e+00]	 2.0647216e-01


.. parsed-literal::

      78	 1.3173699e+00	 1.4795008e-01	 1.3677525e+00	 1.2518691e-01	  1.2765946e+00 	 2.0226240e-01
      79	 1.3217837e+00	 1.4769815e-01	 1.3723077e+00	 1.2480737e-01	  1.2804589e+00 	 2.0246720e-01


.. parsed-literal::

      80	 1.3258541e+00	 1.4736345e-01	 1.3765744e+00	 1.2432448e-01	[ 1.2843690e+00]	 2.0049739e-01
      81	 1.3308482e+00	 1.4707657e-01	 1.3816547e+00	 1.2391129e-01	[ 1.2891610e+00]	 1.8740821e-01


.. parsed-literal::

      82	 1.3364428e+00	 1.4636276e-01	 1.3875009e+00	 1.2240831e-01	[ 1.2907527e+00]	 2.0061541e-01
      83	 1.3426384e+00	 1.4621469e-01	 1.3935621e+00	 1.2251519e-01	[ 1.2960315e+00]	 1.8881679e-01


.. parsed-literal::

      84	 1.3462409e+00	 1.4594600e-01	 1.3971689e+00	 1.2228181e-01	[ 1.2975191e+00]	 2.1062684e-01


.. parsed-literal::

      85	 1.3516233e+00	 1.4542249e-01	 1.4028050e+00	 1.2172871e-01	  1.2969537e+00 	 2.1425843e-01


.. parsed-literal::

      86	 1.3550776e+00	 1.4474597e-01	 1.4064905e+00	 1.2100118e-01	[ 1.2980825e+00]	 3.2855129e-01
      87	 1.3598889e+00	 1.4398652e-01	 1.4115680e+00	 1.2024315e-01	[ 1.2985162e+00]	 1.7772794e-01


.. parsed-literal::

      88	 1.3645038e+00	 1.4318296e-01	 1.4163534e+00	 1.1948365e-01	[ 1.3004296e+00]	 1.9904184e-01
      89	 1.3687777e+00	 1.4264415e-01	 1.4206332e+00	 1.1918433e-01	[ 1.3046838e+00]	 1.8001509e-01


.. parsed-literal::

      90	 1.3729097e+00	 1.4236893e-01	 1.4246053e+00	 1.1910334e-01	[ 1.3084881e+00]	 2.0678544e-01


.. parsed-literal::

      91	 1.3781379e+00	 1.4182976e-01	 1.4297183e+00	 1.1875067e-01	[ 1.3111637e+00]	 2.0367312e-01


.. parsed-literal::

      92	 1.3808949e+00	 1.4160879e-01	 1.4325535e+00	 1.1844350e-01	  1.3109645e+00 	 2.1668744e-01


.. parsed-literal::

      93	 1.3844854e+00	 1.4120615e-01	 1.4361521e+00	 1.1803976e-01	[ 1.3129533e+00]	 2.0837140e-01
      94	 1.3872118e+00	 1.4078125e-01	 1.4390650e+00	 1.1761386e-01	  1.3128692e+00 	 1.9482899e-01


.. parsed-literal::

      95	 1.3902228e+00	 1.4031532e-01	 1.4422704e+00	 1.1714527e-01	[ 1.3145960e+00]	 2.1326184e-01


.. parsed-literal::

      96	 1.3952210e+00	 1.3955166e-01	 1.4475465e+00	 1.1682525e-01	[ 1.3151643e+00]	 2.1536660e-01
      97	 1.3978541e+00	 1.3884539e-01	 1.4504069e+00	 1.1645335e-01	[ 1.3228425e+00]	 1.6836905e-01


.. parsed-literal::

      98	 1.4020673e+00	 1.3882385e-01	 1.4543475e+00	 1.1659434e-01	[ 1.3248344e+00]	 2.1247220e-01
      99	 1.4042313e+00	 1.3881275e-01	 1.4563895e+00	 1.1679416e-01	[ 1.3266920e+00]	 1.8885517e-01


.. parsed-literal::

     100	 1.4079280e+00	 1.3856109e-01	 1.4600647e+00	 1.1691520e-01	[ 1.3287238e+00]	 2.1207476e-01


.. parsed-literal::

     101	 1.4098737e+00	 1.3831148e-01	 1.4622022e+00	 1.1763068e-01	  1.3283052e+00 	 2.0365191e-01
     102	 1.4146504e+00	 1.3778291e-01	 1.4670211e+00	 1.1704484e-01	[ 1.3313241e+00]	 1.8533182e-01


.. parsed-literal::

     103	 1.4168518e+00	 1.3746606e-01	 1.4692873e+00	 1.1661014e-01	[ 1.3332878e+00]	 1.7530847e-01
     104	 1.4193821e+00	 1.3701924e-01	 1.4719459e+00	 1.1631755e-01	[ 1.3345618e+00]	 1.8995190e-01


.. parsed-literal::

     105	 1.4221377e+00	 1.3665567e-01	 1.4747085e+00	 1.1619013e-01	[ 1.3371180e+00]	 2.1598244e-01
     106	 1.4255735e+00	 1.3629815e-01	 1.4780916e+00	 1.1630558e-01	[ 1.3373907e+00]	 1.9963884e-01


.. parsed-literal::

     107	 1.4288997e+00	 1.3624362e-01	 1.4812704e+00	 1.1663003e-01	[ 1.3393594e+00]	 2.1285391e-01


.. parsed-literal::

     108	 1.4318847e+00	 1.3608443e-01	 1.4841141e+00	 1.1666556e-01	  1.3383843e+00 	 2.0555758e-01


.. parsed-literal::

     109	 1.4350081e+00	 1.3607782e-01	 1.4872201e+00	 1.1662805e-01	[ 1.3401753e+00]	 2.1285844e-01
     110	 1.4387756e+00	 1.3589892e-01	 1.4911754e+00	 1.1630148e-01	  1.3368616e+00 	 1.8086529e-01


.. parsed-literal::

     111	 1.4414139e+00	 1.3558752e-01	 1.4941444e+00	 1.1569433e-01	  1.3346019e+00 	 2.1390533e-01


.. parsed-literal::

     112	 1.4438803e+00	 1.3532973e-01	 1.4967079e+00	 1.1556016e-01	  1.3309836e+00 	 2.0886946e-01


.. parsed-literal::

     113	 1.4464556e+00	 1.3504757e-01	 1.4994069e+00	 1.1541547e-01	  1.3276840e+00 	 2.0837021e-01


.. parsed-literal::

     114	 1.4490476e+00	 1.3479870e-01	 1.5020888e+00	 1.1523102e-01	  1.3250664e+00 	 2.1099210e-01


.. parsed-literal::

     115	 1.4500167e+00	 1.3513789e-01	 1.5033834e+00	 1.1544118e-01	  1.3163240e+00 	 2.1731091e-01


.. parsed-literal::

     116	 1.4543299e+00	 1.3483616e-01	 1.5074622e+00	 1.1500436e-01	  1.3239761e+00 	 2.1272373e-01


.. parsed-literal::

     117	 1.4553361e+00	 1.3483055e-01	 1.5084054e+00	 1.1493279e-01	  1.3275099e+00 	 2.1987557e-01


.. parsed-literal::

     118	 1.4572374e+00	 1.3487851e-01	 1.5103028e+00	 1.1475510e-01	  1.3296305e+00 	 2.1925664e-01


.. parsed-literal::

     119	 1.4592558e+00	 1.3485547e-01	 1.5124544e+00	 1.1472284e-01	  1.3317838e+00 	 2.0657015e-01


.. parsed-literal::

     120	 1.4614201e+00	 1.3482067e-01	 1.5146943e+00	 1.1453435e-01	  1.3294270e+00 	 2.1087003e-01
     121	 1.4634394e+00	 1.3473601e-01	 1.5168431e+00	 1.1443578e-01	  1.3254048e+00 	 1.8687177e-01


.. parsed-literal::

     122	 1.4651190e+00	 1.3461615e-01	 1.5185620e+00	 1.1437319e-01	  1.3250214e+00 	 2.1395946e-01


.. parsed-literal::

     123	 1.4681355e+00	 1.3444261e-01	 1.5215468e+00	 1.1439974e-01	  1.3266683e+00 	 2.0474601e-01


.. parsed-literal::

     124	 1.4695996e+00	 1.3406850e-01	 1.5229752e+00	 1.1399229e-01	  1.3283674e+00 	 2.1280408e-01


.. parsed-literal::

     125	 1.4720789e+00	 1.3407966e-01	 1.5252853e+00	 1.1413515e-01	  1.3336491e+00 	 2.1186996e-01
     126	 1.4735496e+00	 1.3408986e-01	 1.5266386e+00	 1.1416018e-01	  1.3364119e+00 	 1.9681144e-01


.. parsed-literal::

     127	 1.4756839e+00	 1.3405433e-01	 1.5286739e+00	 1.1410589e-01	  1.3382168e+00 	 2.2131515e-01
     128	 1.4785186e+00	 1.3405349e-01	 1.5315321e+00	 1.1382240e-01	[ 1.3422106e+00]	 1.9428968e-01


.. parsed-literal::

     129	 1.4805896e+00	 1.3398685e-01	 1.5337111e+00	 1.1341772e-01	  1.3379162e+00 	 2.0368457e-01


.. parsed-literal::

     130	 1.4824128e+00	 1.3400613e-01	 1.5355737e+00	 1.1341221e-01	  1.3394372e+00 	 2.1653581e-01


.. parsed-literal::

     131	 1.4835598e+00	 1.3398904e-01	 1.5368007e+00	 1.1328932e-01	  1.3393301e+00 	 2.1120811e-01


.. parsed-literal::

     132	 1.4849772e+00	 1.3394533e-01	 1.5383477e+00	 1.1321431e-01	  1.3380243e+00 	 2.1807766e-01
     133	 1.4869524e+00	 1.3393687e-01	 1.5404076e+00	 1.1316694e-01	  1.3323143e+00 	 1.8630624e-01


.. parsed-literal::

     134	 1.4891158e+00	 1.3375814e-01	 1.5425738e+00	 1.1321693e-01	  1.3266978e+00 	 1.8832731e-01


.. parsed-literal::

     135	 1.4908947e+00	 1.3361579e-01	 1.5443268e+00	 1.1332181e-01	  1.3226864e+00 	 2.1926188e-01
     136	 1.4927441e+00	 1.3338870e-01	 1.5461645e+00	 1.1347808e-01	  1.3202292e+00 	 1.8406534e-01


.. parsed-literal::

     137	 1.4945890e+00	 1.3320820e-01	 1.5480564e+00	 1.1369155e-01	  1.3191535e+00 	 2.1050215e-01


.. parsed-literal::

     138	 1.4962838e+00	 1.3305082e-01	 1.5498390e+00	 1.1361036e-01	  1.3174450e+00 	 2.0654035e-01


.. parsed-literal::

     139	 1.4975491e+00	 1.3299203e-01	 1.5511700e+00	 1.1354043e-01	  1.3169455e+00 	 2.2017264e-01


.. parsed-literal::

     140	 1.4996579e+00	 1.3285024e-01	 1.5534397e+00	 1.1338885e-01	  1.3127136e+00 	 2.1108770e-01


.. parsed-literal::

     141	 1.5018593e+00	 1.3283380e-01	 1.5556913e+00	 1.1348050e-01	  1.3139518e+00 	 2.0388818e-01


.. parsed-literal::

     142	 1.5036316e+00	 1.3279752e-01	 1.5574311e+00	 1.1355503e-01	  1.3139220e+00 	 2.0140862e-01
     143	 1.5061469e+00	 1.3274212e-01	 1.5599190e+00	 1.1363694e-01	  1.3203377e+00 	 1.7473531e-01


.. parsed-literal::

     144	 1.5076408e+00	 1.3260232e-01	 1.5613535e+00	 1.1359438e-01	  1.3196075e+00 	 2.0309138e-01
     145	 1.5089295e+00	 1.3260356e-01	 1.5626023e+00	 1.1355076e-01	  1.3232391e+00 	 1.9495630e-01


.. parsed-literal::

     146	 1.5098335e+00	 1.3258913e-01	 1.5635251e+00	 1.1350960e-01	  1.3258860e+00 	 2.0268869e-01


.. parsed-literal::

     147	 1.5107857e+00	 1.3253941e-01	 1.5645075e+00	 1.1343189e-01	  1.3288424e+00 	 2.1023130e-01


.. parsed-literal::

     148	 1.5120008e+00	 1.3251640e-01	 1.5657692e+00	 1.1330885e-01	  1.3301962e+00 	 2.0224905e-01


.. parsed-literal::

     149	 1.5131978e+00	 1.3247378e-01	 1.5670064e+00	 1.1322651e-01	  1.3306091e+00 	 2.2128057e-01


.. parsed-literal::

     150	 1.5145805e+00	 1.3238304e-01	 1.5684979e+00	 1.1311763e-01	  1.3277988e+00 	 2.1483016e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.11 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4e2042f7c0>



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
    CPU times: user 2.01 s, sys: 39 ms, total: 2.05 s
    Wall time: 612 ms


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

