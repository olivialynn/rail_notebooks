Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f18f2d440a0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.014542  0.008191  
    1      25.391064  0.157186  0.125344  
    2      24.304707  0.000548  0.000382  
    3      25.291103  0.046234  0.034181  
    4      25.096743  0.053809  0.043935  
    ...          ...       ...       ...  
    99995  24.737946  0.085726  0.084596  
    99996  24.224169  0.054463  0.036147  
    99997  25.613836  0.198316  0.155870  
    99998  25.274899  0.009316  0.005827  
    99999  25.699642  0.005304  0.003078  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.603032</td>
          <td>0.408230</td>
          <td>26.748505</td>
          <td>0.171321</td>
          <td>25.854730</td>
          <td>0.069343</td>
          <td>25.145330</td>
          <td>0.060396</td>
          <td>24.557956</td>
          <td>0.068720</td>
          <td>23.973949</td>
          <td>0.092374</td>
          <td>0.014542</td>
          <td>0.008191</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.904562</td>
          <td>0.511880</td>
          <td>27.248693</td>
          <td>0.260111</td>
          <td>26.595728</td>
          <td>0.132774</td>
          <td>26.295659</td>
          <td>0.165073</td>
          <td>25.647743</td>
          <td>0.177449</td>
          <td>25.341851</td>
          <td>0.294956</td>
          <td>0.157186</td>
          <td>0.125344</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.153592</td>
          <td>0.612200</td>
          <td>28.289793</td>
          <td>0.579711</td>
          <td>28.289735</td>
          <td>0.522512</td>
          <td>25.800152</td>
          <td>0.107584</td>
          <td>24.893974</td>
          <td>0.092436</td>
          <td>24.484509</td>
          <td>0.144056</td>
          <td>0.000548</td>
          <td>0.000382</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.875823</td>
          <td>0.501178</td>
          <td>28.885685</td>
          <td>0.866552</td>
          <td>27.461791</td>
          <td>0.275171</td>
          <td>26.149120</td>
          <td>0.145603</td>
          <td>25.403176</td>
          <td>0.143983</td>
          <td>25.968177</td>
          <td>0.479721</td>
          <td>0.046234</td>
          <td>0.034181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.648453</td>
          <td>0.188774</td>
          <td>26.123517</td>
          <td>0.099851</td>
          <td>25.908338</td>
          <td>0.072712</td>
          <td>25.883023</td>
          <td>0.115648</td>
          <td>25.295412</td>
          <td>0.131198</td>
          <td>25.000195</td>
          <td>0.222952</td>
          <td>0.053809</td>
          <td>0.043935</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.779109</td>
          <td>0.926309</td>
          <td>26.321381</td>
          <td>0.118657</td>
          <td>25.435274</td>
          <td>0.047795</td>
          <td>25.141213</td>
          <td>0.060176</td>
          <td>24.791071</td>
          <td>0.084435</td>
          <td>25.093829</td>
          <td>0.240932</td>
          <td>0.085726</td>
          <td>0.084596</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.114121</td>
          <td>0.277488</td>
          <td>26.899451</td>
          <td>0.194650</td>
          <td>26.282565</td>
          <td>0.101092</td>
          <td>25.092887</td>
          <td>0.057650</td>
          <td>24.802268</td>
          <td>0.085272</td>
          <td>24.373612</td>
          <td>0.130910</td>
          <td>0.054463</td>
          <td>0.036147</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.290991</td>
          <td>0.319866</td>
          <td>27.156329</td>
          <td>0.241109</td>
          <td>26.184225</td>
          <td>0.092737</td>
          <td>26.132099</td>
          <td>0.143487</td>
          <td>25.995375</td>
          <td>0.237436</td>
          <td>25.923947</td>
          <td>0.464137</td>
          <td>0.198316</td>
          <td>0.155870</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.383064</td>
          <td>0.344064</td>
          <td>26.066393</td>
          <td>0.094978</td>
          <td>26.071611</td>
          <td>0.083988</td>
          <td>25.823062</td>
          <td>0.109758</td>
          <td>25.809664</td>
          <td>0.203419</td>
          <td>25.355121</td>
          <td>0.298125</td>
          <td>0.009316</td>
          <td>0.005827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.879812</td>
          <td>0.502653</td>
          <td>26.861333</td>
          <td>0.188498</td>
          <td>26.574851</td>
          <td>0.130397</td>
          <td>26.025492</td>
          <td>0.130874</td>
          <td>25.691172</td>
          <td>0.184098</td>
          <td>26.229542</td>
          <td>0.580401</td>
          <td>0.005304</td>
          <td>0.003078</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.916744</td>
          <td>0.569048</td>
          <td>26.661297</td>
          <td>0.182721</td>
          <td>25.879224</td>
          <td>0.083415</td>
          <td>25.171168</td>
          <td>0.073287</td>
          <td>24.679800</td>
          <td>0.090050</td>
          <td>23.984538</td>
          <td>0.110179</td>
          <td>0.014542</td>
          <td>0.008191</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.073514</td>
          <td>1.995329</td>
          <td>27.667876</td>
          <td>0.433955</td>
          <td>26.680522</td>
          <td>0.177923</td>
          <td>26.272040</td>
          <td>0.202825</td>
          <td>25.667819</td>
          <td>0.223870</td>
          <td>24.936621</td>
          <td>0.263188</td>
          <td>0.157186</td>
          <td>0.125344</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.803879</td>
          <td>1.584651</td>
          <td>27.491960</td>
          <td>0.326803</td>
          <td>26.065515</td>
          <td>0.159737</td>
          <td>25.054093</td>
          <td>0.124816</td>
          <td>24.336923</td>
          <td>0.149407</td>
          <td>0.000548</td>
          <td>0.000382</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.443252</td>
          <td>0.401577</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.125207</td>
          <td>0.532323</td>
          <td>26.428827</td>
          <td>0.218299</td>
          <td>25.176270</td>
          <td>0.139488</td>
          <td>25.238634</td>
          <td>0.317843</td>
          <td>0.046234</td>
          <td>0.034181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.321940</td>
          <td>0.366194</td>
          <td>26.052102</td>
          <td>0.108957</td>
          <td>25.960723</td>
          <td>0.090311</td>
          <td>25.588821</td>
          <td>0.106655</td>
          <td>25.765020</td>
          <td>0.230182</td>
          <td>25.656834</td>
          <td>0.441054</td>
          <td>0.053809</td>
          <td>0.043935</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.294477</td>
          <td>0.136372</td>
          <td>25.473381</td>
          <td>0.059667</td>
          <td>25.088295</td>
          <td>0.069808</td>
          <td>24.599311</td>
          <td>0.085905</td>
          <td>24.850236</td>
          <td>0.235911</td>
          <td>0.085726</td>
          <td>0.084596</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.408595</td>
          <td>0.799931</td>
          <td>26.828685</td>
          <td>0.211550</td>
          <td>25.993531</td>
          <td>0.092865</td>
          <td>25.194887</td>
          <td>0.075368</td>
          <td>24.735483</td>
          <td>0.095203</td>
          <td>24.243230</td>
          <td>0.138843</td>
          <td>0.054463</td>
          <td>0.036147</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.770634</td>
          <td>0.544830</td>
          <td>26.648028</td>
          <td>0.196638</td>
          <td>26.379899</td>
          <td>0.142118</td>
          <td>26.255539</td>
          <td>0.206650</td>
          <td>25.924366</td>
          <td>0.284985</td>
          <td>25.529563</td>
          <td>0.433308</td>
          <td>0.198316</td>
          <td>0.155870</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.010521</td>
          <td>0.608142</td>
          <td>26.181870</td>
          <td>0.121110</td>
          <td>25.999391</td>
          <td>0.092691</td>
          <td>25.872410</td>
          <td>0.135346</td>
          <td>25.858784</td>
          <td>0.246865</td>
          <td>26.000842</td>
          <td>0.564566</td>
          <td>0.009316</td>
          <td>0.005827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.223827</td>
          <td>1.294240</td>
          <td>26.643984</td>
          <td>0.180000</td>
          <td>27.103084</td>
          <td>0.238431</td>
          <td>26.836901</td>
          <td>0.303263</td>
          <td>25.971539</td>
          <td>0.270700</td>
          <td>25.304780</td>
          <td>0.333287</td>
          <td>0.005304</td>
          <td>0.003078</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.245642</td>
          <td>0.653469</td>
          <td>26.884474</td>
          <td>0.192513</td>
          <td>25.983349</td>
          <td>0.077844</td>
          <td>25.129047</td>
          <td>0.059649</td>
          <td>24.698149</td>
          <td>0.077938</td>
          <td>23.924464</td>
          <td>0.088616</td>
          <td>0.014542</td>
          <td>0.008191</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.124723</td>
          <td>0.280888</td>
          <td>26.822989</td>
          <td>0.198695</td>
          <td>26.270258</td>
          <td>0.200482</td>
          <td>25.737466</td>
          <td>0.234899</td>
          <td>26.051740</td>
          <td>0.612033</td>
          <td>0.157186</td>
          <td>0.125344</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.957885</td>
          <td>1.744820</td>
          <td>29.458954</td>
          <td>1.218080</td>
          <td>29.468015</td>
          <td>1.129178</td>
          <td>26.052715</td>
          <td>0.133992</td>
          <td>24.998802</td>
          <td>0.101339</td>
          <td>24.185666</td>
          <td>0.111188</td>
          <td>0.000548</td>
          <td>0.000382</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.414842</td>
          <td>0.642802</td>
          <td>27.529858</td>
          <td>0.296675</td>
          <td>26.359199</td>
          <td>0.178159</td>
          <td>25.699940</td>
          <td>0.189412</td>
          <td>25.079372</td>
          <td>0.243217</td>
          <td>0.046234</td>
          <td>0.034181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.358024</td>
          <td>0.344113</td>
          <td>26.332068</td>
          <td>0.123076</td>
          <td>25.915068</td>
          <td>0.075519</td>
          <td>25.677505</td>
          <td>0.099891</td>
          <td>25.557344</td>
          <td>0.169437</td>
          <td>25.163905</td>
          <td>0.263167</td>
          <td>0.053809</td>
          <td>0.043935</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.874787</td>
          <td>0.528140</td>
          <td>26.200422</td>
          <td>0.115467</td>
          <td>25.485485</td>
          <td>0.054725</td>
          <td>25.157045</td>
          <td>0.067093</td>
          <td>24.798433</td>
          <td>0.092964</td>
          <td>25.104060</td>
          <td>0.264973</td>
          <td>0.085726</td>
          <td>0.084596</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.674595</td>
          <td>0.879920</td>
          <td>26.834544</td>
          <td>0.188663</td>
          <td>26.015076</td>
          <td>0.082190</td>
          <td>25.400706</td>
          <td>0.078003</td>
          <td>24.844400</td>
          <td>0.091016</td>
          <td>24.029268</td>
          <td>0.099835</td>
          <td>0.054463</td>
          <td>0.036147</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.753222</td>
          <td>0.554799</td>
          <td>26.435851</td>
          <td>0.171254</td>
          <td>26.381557</td>
          <td>0.148893</td>
          <td>26.161852</td>
          <td>0.199860</td>
          <td>25.818449</td>
          <td>0.272884</td>
          <td>26.006696</td>
          <td>0.637824</td>
          <td>0.198316</td>
          <td>0.155870</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.810274</td>
          <td>0.216281</td>
          <td>26.098003</td>
          <td>0.097715</td>
          <td>26.049788</td>
          <td>0.082455</td>
          <td>25.997227</td>
          <td>0.127818</td>
          <td>25.722863</td>
          <td>0.189242</td>
          <td>26.383552</td>
          <td>0.647257</td>
          <td>0.009316</td>
          <td>0.005827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.451966</td>
          <td>0.363235</td>
          <td>26.802305</td>
          <td>0.179361</td>
          <td>26.600579</td>
          <td>0.133365</td>
          <td>26.361874</td>
          <td>0.174688</td>
          <td>26.332452</td>
          <td>0.312418</td>
          <td>26.204634</td>
          <td>0.570283</td>
          <td>0.005304</td>
          <td>0.003078</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
