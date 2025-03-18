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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f066ecd7fa0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>28.220486</td>
          <td>1.200878</td>
          <td>26.724543</td>
          <td>0.167865</td>
          <td>26.071926</td>
          <td>0.084011</td>
          <td>25.149845</td>
          <td>0.060638</td>
          <td>24.641950</td>
          <td>0.074021</td>
          <td>24.074389</td>
          <td>0.100883</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.945466</td>
          <td>1.024831</td>
          <td>28.168370</td>
          <td>0.531137</td>
          <td>26.540353</td>
          <td>0.126559</td>
          <td>26.416221</td>
          <td>0.182877</td>
          <td>25.889250</td>
          <td>0.217417</td>
          <td>25.077773</td>
          <td>0.237760</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.325423</td>
          <td>0.689546</td>
          <td>31.407092</td>
          <td>2.822127</td>
          <td>27.747960</td>
          <td>0.346045</td>
          <td>25.847264</td>
          <td>0.112100</td>
          <td>24.990344</td>
          <td>0.100591</td>
          <td>24.341686</td>
          <td>0.127341</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.657558</td>
          <td>0.747118</td>
          <td>27.404811</td>
          <td>0.262684</td>
          <td>26.004894</td>
          <td>0.128561</td>
          <td>25.330867</td>
          <td>0.135282</td>
          <td>25.360776</td>
          <td>0.299484</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.964119</td>
          <td>0.245503</td>
          <td>26.068933</td>
          <td>0.095190</td>
          <td>25.884040</td>
          <td>0.071165</td>
          <td>25.642956</td>
          <td>0.093744</td>
          <td>25.425180</td>
          <td>0.146733</td>
          <td>25.438057</td>
          <td>0.318607</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.280310</td>
          <td>1.241258</td>
          <td>26.287096</td>
          <td>0.115172</td>
          <td>25.453709</td>
          <td>0.048584</td>
          <td>25.032257</td>
          <td>0.054629</td>
          <td>24.727599</td>
          <td>0.079839</td>
          <td>24.914822</td>
          <td>0.207623</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.469763</td>
          <td>0.368259</td>
          <td>26.695291</td>
          <td>0.163734</td>
          <td>26.289087</td>
          <td>0.101672</td>
          <td>25.117081</td>
          <td>0.058901</td>
          <td>24.935015</td>
          <td>0.095828</td>
          <td>24.197188</td>
          <td>0.112310</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.187775</td>
          <td>0.294499</td>
          <td>26.647886</td>
          <td>0.157239</td>
          <td>26.327620</td>
          <td>0.105158</td>
          <td>26.030278</td>
          <td>0.131417</td>
          <td>25.889706</td>
          <td>0.217499</td>
          <td>25.464577</td>
          <td>0.325408</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.723476</td>
          <td>0.447375</td>
          <td>26.215207</td>
          <td>0.108180</td>
          <td>26.237529</td>
          <td>0.097180</td>
          <td>25.914756</td>
          <td>0.118887</td>
          <td>25.959617</td>
          <td>0.230513</td>
          <td>25.000588</td>
          <td>0.223025</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.086591</td>
          <td>0.271355</td>
          <td>26.997832</td>
          <td>0.211386</td>
          <td>26.525227</td>
          <td>0.124909</td>
          <td>26.231354</td>
          <td>0.156248</td>
          <td>26.260630</td>
          <td>0.294844</td>
          <td>26.144907</td>
          <td>0.546163</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.624033</td>
          <td>0.176979</td>
          <td>26.074508</td>
          <td>0.098988</td>
          <td>25.118048</td>
          <td>0.069891</td>
          <td>24.752800</td>
          <td>0.095969</td>
          <td>24.037374</td>
          <td>0.115318</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.789912</td>
          <td>1.011472</td>
          <td>26.937506</td>
          <td>0.230214</td>
          <td>26.672871</td>
          <td>0.166166</td>
          <td>26.093575</td>
          <td>0.163649</td>
          <td>26.081389</td>
          <td>0.295935</td>
          <td>25.247214</td>
          <td>0.318429</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.981824</td>
          <td>0.604106</td>
          <td>28.059826</td>
          <td>0.561524</td>
          <td>28.152218</td>
          <td>0.550391</td>
          <td>25.786721</td>
          <td>0.128544</td>
          <td>25.033685</td>
          <td>0.125359</td>
          <td>24.148789</td>
          <td>0.129936</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.745733</td>
          <td>1.020309</td>
          <td>28.843488</td>
          <td>0.974803</td>
          <td>27.109828</td>
          <td>0.255305</td>
          <td>26.071990</td>
          <td>0.171807</td>
          <td>25.536407</td>
          <td>0.201171</td>
          <td>26.061003</td>
          <td>0.622756</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.384934</td>
          <td>0.382563</td>
          <td>26.052411</td>
          <td>0.108227</td>
          <td>25.972685</td>
          <td>0.090555</td>
          <td>25.690488</td>
          <td>0.115612</td>
          <td>25.443085</td>
          <td>0.174374</td>
          <td>25.585646</td>
          <td>0.414916</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.605086</td>
          <td>0.458746</td>
          <td>26.430596</td>
          <td>0.152875</td>
          <td>25.427209</td>
          <td>0.057095</td>
          <td>25.025738</td>
          <td>0.065836</td>
          <td>24.876613</td>
          <td>0.109209</td>
          <td>24.495847</td>
          <td>0.174748</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.356523</td>
          <td>0.771786</td>
          <td>26.883988</td>
          <td>0.220957</td>
          <td>26.162906</td>
          <td>0.107391</td>
          <td>25.106902</td>
          <td>0.069508</td>
          <td>24.910506</td>
          <td>0.110622</td>
          <td>24.276829</td>
          <td>0.142487</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.868304</td>
          <td>0.553828</td>
          <td>26.774940</td>
          <td>0.203225</td>
          <td>26.332787</td>
          <td>0.125569</td>
          <td>26.629964</td>
          <td>0.259564</td>
          <td>26.048141</td>
          <td>0.291448</td>
          <td>25.009613</td>
          <td>0.266010</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.996387</td>
          <td>0.613755</td>
          <td>26.119420</td>
          <td>0.117972</td>
          <td>26.231888</td>
          <td>0.117174</td>
          <td>25.819944</td>
          <td>0.133526</td>
          <td>25.427253</td>
          <td>0.177288</td>
          <td>25.457169</td>
          <td>0.386502</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.171629</td>
          <td>1.264113</td>
          <td>26.972308</td>
          <td>0.238921</td>
          <td>26.947808</td>
          <td>0.211565</td>
          <td>26.685312</td>
          <td>0.270844</td>
          <td>26.174825</td>
          <td>0.321799</td>
          <td>25.469305</td>
          <td>0.382678</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.531081</td>
          <td>0.142264</td>
          <td>25.988933</td>
          <td>0.078090</td>
          <td>25.275358</td>
          <td>0.067784</td>
          <td>24.554044</td>
          <td>0.068492</td>
          <td>23.905410</td>
          <td>0.086983</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.251245</td>
          <td>1.985967</td>
          <td>27.246741</td>
          <td>0.259894</td>
          <td>26.553131</td>
          <td>0.128087</td>
          <td>26.581995</td>
          <td>0.210447</td>
          <td>25.651774</td>
          <td>0.178219</td>
          <td>25.888337</td>
          <td>0.452269</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.468283</td>
          <td>1.280336</td>
          <td>29.670595</td>
          <td>1.329225</td>
          <td>26.160801</td>
          <td>0.159870</td>
          <td>25.109560</td>
          <td>0.121062</td>
          <td>24.357124</td>
          <td>0.140272</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.197458</td>
          <td>0.718742</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.178574</td>
          <td>0.580117</td>
          <td>26.024714</td>
          <td>0.164441</td>
          <td>25.765014</td>
          <td>0.242494</td>
          <td>24.947521</td>
          <td>0.265397</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.593030</td>
          <td>0.405458</td>
          <td>26.319742</td>
          <td>0.118632</td>
          <td>25.979368</td>
          <td>0.077534</td>
          <td>25.846544</td>
          <td>0.112196</td>
          <td>25.421554</td>
          <td>0.146480</td>
          <td>24.636880</td>
          <td>0.164381</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.582366</td>
          <td>0.421837</td>
          <td>26.923315</td>
          <td>0.212090</td>
          <td>25.515998</td>
          <td>0.055609</td>
          <td>25.098353</td>
          <td>0.062964</td>
          <td>24.839935</td>
          <td>0.095367</td>
          <td>24.513199</td>
          <td>0.159882</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.625532</td>
          <td>2.319004</td>
          <td>26.571344</td>
          <td>0.149333</td>
          <td>26.213718</td>
          <td>0.096751</td>
          <td>25.184263</td>
          <td>0.063620</td>
          <td>24.865433</td>
          <td>0.091644</td>
          <td>24.347464</td>
          <td>0.130157</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.494433</td>
          <td>0.792326</td>
          <td>26.603620</td>
          <td>0.157829</td>
          <td>26.400981</td>
          <td>0.117688</td>
          <td>26.270590</td>
          <td>0.169815</td>
          <td>25.822578</td>
          <td>0.215418</td>
          <td>27.072954</td>
          <td>1.045894</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.812503</td>
          <td>0.512422</td>
          <td>26.230197</td>
          <td>0.121133</td>
          <td>26.245969</td>
          <td>0.109751</td>
          <td>25.855570</td>
          <td>0.127126</td>
          <td>25.486658</td>
          <td>0.172880</td>
          <td>25.601891</td>
          <td>0.402678</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.590229</td>
          <td>0.413967</td>
          <td>27.031532</td>
          <td>0.224507</td>
          <td>26.424797</td>
          <td>0.118967</td>
          <td>26.181632</td>
          <td>0.155808</td>
          <td>25.447709</td>
          <td>0.155384</td>
          <td>25.558313</td>
          <td>0.363311</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
