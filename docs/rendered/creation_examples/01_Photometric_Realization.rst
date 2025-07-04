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

    <pzflow.flow.Flow at 0x7fe394b74be0>



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
    0      23.994413  0.084678  0.066187  
    1      25.391064  0.038112  0.035985  
    2      24.304707  0.141438  0.086934  
    3      25.291103  0.132965  0.083590  
    4      25.096743  0.054936  0.048589  
    ...          ...       ...       ...  
    99995  24.737946  0.087648  0.067681  
    99996  24.224169  0.093276  0.091338  
    99997  25.613836  0.049838  0.028235  
    99998  25.274899  0.027270  0.018626  
    99999  25.699642  0.012691  0.006402  
    
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
          <td>27.934757</td>
          <td>1.018304</td>
          <td>26.841983</td>
          <td>0.185443</td>
          <td>25.924096</td>
          <td>0.073732</td>
          <td>25.183214</td>
          <td>0.062460</td>
          <td>24.881975</td>
          <td>0.091467</td>
          <td>23.959430</td>
          <td>0.091203</td>
          <td>0.084678</td>
          <td>0.066187</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.816962</td>
          <td>0.408370</td>
          <td>26.538455</td>
          <td>0.126351</td>
          <td>26.068153</td>
          <td>0.135790</td>
          <td>25.644971</td>
          <td>0.177032</td>
          <td>25.552472</td>
          <td>0.348845</td>
          <td>0.038112</td>
          <td>0.035985</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.621895</td>
          <td>2.119369</td>
          <td>28.581054</td>
          <td>0.643150</td>
          <td>25.879769</td>
          <td>0.115321</td>
          <td>25.224993</td>
          <td>0.123432</td>
          <td>24.376767</td>
          <td>0.131268</td>
          <td>0.141438</td>
          <td>0.086934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.339785</td>
          <td>0.696317</td>
          <td>29.025706</td>
          <td>0.945743</td>
          <td>27.720664</td>
          <td>0.338668</td>
          <td>26.061246</td>
          <td>0.134983</td>
          <td>25.643319</td>
          <td>0.176785</td>
          <td>24.929453</td>
          <td>0.210180</td>
          <td>0.132965</td>
          <td>0.083590</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.117718</td>
          <td>0.278298</td>
          <td>25.998406</td>
          <td>0.089477</td>
          <td>25.764181</td>
          <td>0.063997</td>
          <td>25.729527</td>
          <td>0.101139</td>
          <td>25.579455</td>
          <td>0.167442</td>
          <td>25.594293</td>
          <td>0.360491</td>
          <td>0.054936</td>
          <td>0.048589</td>
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
          <td>27.124802</td>
          <td>0.599896</td>
          <td>26.639287</td>
          <td>0.156086</td>
          <td>25.351824</td>
          <td>0.044382</td>
          <td>25.056887</td>
          <td>0.055836</td>
          <td>24.885547</td>
          <td>0.091754</td>
          <td>24.439370</td>
          <td>0.138562</td>
          <td>0.087648</td>
          <td>0.067681</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.642589</td>
          <td>0.156528</td>
          <td>26.017136</td>
          <td>0.080048</td>
          <td>25.181222</td>
          <td>0.062350</td>
          <td>24.797764</td>
          <td>0.084934</td>
          <td>24.078049</td>
          <td>0.101206</td>
          <td>0.093276</td>
          <td>0.091338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.601003</td>
          <td>1.469554</td>
          <td>27.077669</td>
          <td>0.225917</td>
          <td>26.382051</td>
          <td>0.110279</td>
          <td>26.295350</td>
          <td>0.165030</td>
          <td>25.816546</td>
          <td>0.204597</td>
          <td>25.038682</td>
          <td>0.230191</td>
          <td>0.049838</td>
          <td>0.028235</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.621841</td>
          <td>0.184587</td>
          <td>26.151979</td>
          <td>0.102368</td>
          <td>26.135069</td>
          <td>0.088815</td>
          <td>25.784069</td>
          <td>0.106083</td>
          <td>25.437440</td>
          <td>0.148287</td>
          <td>25.256986</td>
          <td>0.275378</td>
          <td>0.027270</td>
          <td>0.018626</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.990650</td>
          <td>0.545010</td>
          <td>26.479177</td>
          <td>0.136027</td>
          <td>26.468453</td>
          <td>0.118900</td>
          <td>26.124246</td>
          <td>0.142520</td>
          <td>25.577361</td>
          <td>0.167143</td>
          <td>25.188619</td>
          <td>0.260446</td>
          <td>0.012691</td>
          <td>0.006402</td>
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
          <td>26.576537</td>
          <td>0.172868</td>
          <td>25.732428</td>
          <td>0.074680</td>
          <td>25.187585</td>
          <td>0.075829</td>
          <td>24.616801</td>
          <td>0.086816</td>
          <td>24.002867</td>
          <td>0.114121</td>
          <td>0.084678</td>
          <td>0.066187</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.830958</td>
          <td>1.038872</td>
          <td>28.196306</td>
          <td>0.610827</td>
          <td>26.640830</td>
          <td>0.162382</td>
          <td>26.250313</td>
          <td>0.187765</td>
          <td>25.582530</td>
          <td>0.196995</td>
          <td>25.723623</td>
          <td>0.462382</td>
          <td>0.038112</td>
          <td>0.035985</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.592568</td>
          <td>0.918127</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.618987</td>
          <td>0.375779</td>
          <td>26.037371</td>
          <td>0.163061</td>
          <td>25.033108</td>
          <td>0.128056</td>
          <td>24.072798</td>
          <td>0.124408</td>
          <td>0.141438</td>
          <td>0.086934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.298060</td>
          <td>0.290171</td>
          <td>26.358029</td>
          <td>0.212844</td>
          <td>25.291049</td>
          <td>0.159190</td>
          <td>25.044083</td>
          <td>0.280728</td>
          <td>0.132965</td>
          <td>0.083590</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.956873</td>
          <td>0.273968</td>
          <td>26.103660</td>
          <td>0.114055</td>
          <td>25.902812</td>
          <td>0.085905</td>
          <td>25.603399</td>
          <td>0.108125</td>
          <td>25.826023</td>
          <td>0.242298</td>
          <td>24.781337</td>
          <td>0.219580</td>
          <td>0.054936</td>
          <td>0.048589</td>
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
          <td>26.486537</td>
          <td>0.160280</td>
          <td>25.414042</td>
          <td>0.056398</td>
          <td>25.122340</td>
          <td>0.071667</td>
          <td>24.948043</td>
          <td>0.116154</td>
          <td>24.876745</td>
          <td>0.240279</td>
          <td>0.087648</td>
          <td>0.067681</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.245410</td>
          <td>0.726851</td>
          <td>26.519812</td>
          <td>0.166023</td>
          <td>26.054065</td>
          <td>0.100003</td>
          <td>25.238530</td>
          <td>0.080057</td>
          <td>24.899380</td>
          <td>0.112195</td>
          <td>24.572718</td>
          <td>0.187827</td>
          <td>0.093276</td>
          <td>0.091338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.378250</td>
          <td>0.783485</td>
          <td>27.196820</td>
          <td>0.285943</td>
          <td>26.531458</td>
          <td>0.147994</td>
          <td>26.509835</td>
          <td>0.233479</td>
          <td>26.319215</td>
          <td>0.359225</td>
          <td>25.504510</td>
          <td>0.391648</td>
          <td>0.049838</td>
          <td>0.028235</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.184772</td>
          <td>0.327317</td>
          <td>26.024084</td>
          <td>0.105727</td>
          <td>26.107634</td>
          <td>0.102086</td>
          <td>25.915192</td>
          <td>0.140669</td>
          <td>26.149142</td>
          <td>0.312931</td>
          <td>25.270209</td>
          <td>0.324804</td>
          <td>0.027270</td>
          <td>0.018626</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.795613</td>
          <td>0.204567</td>
          <td>26.527968</td>
          <td>0.146801</td>
          <td>26.229810</td>
          <td>0.183748</td>
          <td>25.755788</td>
          <td>0.226750</td>
          <td>25.275906</td>
          <td>0.325820</td>
          <td>0.012691</td>
          <td>0.006402</td>
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
          <td>28.814197</td>
          <td>1.677125</td>
          <td>26.687401</td>
          <td>0.172944</td>
          <td>25.956190</td>
          <td>0.081587</td>
          <td>25.112406</td>
          <td>0.063331</td>
          <td>24.656286</td>
          <td>0.080627</td>
          <td>23.853215</td>
          <td>0.089560</td>
          <td>0.084678</td>
          <td>0.066187</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.127335</td>
          <td>0.522327</td>
          <td>26.714774</td>
          <td>0.149781</td>
          <td>26.184570</td>
          <td>0.152950</td>
          <td>25.842420</td>
          <td>0.212749</td>
          <td>25.098485</td>
          <td>0.246213</td>
          <td>0.038112</td>
          <td>0.035985</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.138040</td>
          <td>1.231461</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.129106</td>
          <td>0.167376</td>
          <td>24.910566</td>
          <td>0.109366</td>
          <td>24.061876</td>
          <td>0.116907</td>
          <td>0.141438</td>
          <td>0.086934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.198727</td>
          <td>0.600843</td>
          <td>27.437733</td>
          <td>0.307090</td>
          <td>26.114654</td>
          <td>0.163101</td>
          <td>25.337376</td>
          <td>0.156102</td>
          <td>25.376961</td>
          <td>0.346064</td>
          <td>0.132965</td>
          <td>0.083590</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.113514</td>
          <td>0.283726</td>
          <td>26.027710</td>
          <td>0.094670</td>
          <td>25.875286</td>
          <td>0.073169</td>
          <td>25.626115</td>
          <td>0.095843</td>
          <td>25.431953</td>
          <td>0.152749</td>
          <td>25.024244</td>
          <td>0.235425</td>
          <td>0.054936</td>
          <td>0.048589</td>
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
          <td>27.893069</td>
          <td>1.030526</td>
          <td>26.511374</td>
          <td>0.149340</td>
          <td>25.459533</td>
          <td>0.052772</td>
          <td>24.994601</td>
          <td>0.057294</td>
          <td>25.019507</td>
          <td>0.111344</td>
          <td>24.655251</td>
          <td>0.180022</td>
          <td>0.087648</td>
          <td>0.067681</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.568948</td>
          <td>0.424204</td>
          <td>26.878241</td>
          <td>0.208603</td>
          <td>26.088340</td>
          <td>0.094570</td>
          <td>25.229272</td>
          <td>0.072587</td>
          <td>24.833411</td>
          <td>0.097220</td>
          <td>24.175851</td>
          <td>0.122623</td>
          <td>0.093276</td>
          <td>0.091338</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.308394</td>
          <td>0.689628</td>
          <td>26.632201</td>
          <td>0.158018</td>
          <td>26.348926</td>
          <td>0.109469</td>
          <td>26.260628</td>
          <td>0.163789</td>
          <td>25.691124</td>
          <td>0.187963</td>
          <td>25.662545</td>
          <td>0.387821</td>
          <td>0.049838</td>
          <td>0.028235</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.016697</td>
          <td>0.257540</td>
          <td>26.152071</td>
          <td>0.103029</td>
          <td>26.103776</td>
          <td>0.087041</td>
          <td>26.123651</td>
          <td>0.143528</td>
          <td>25.456657</td>
          <td>0.151838</td>
          <td>25.201033</td>
          <td>0.264977</td>
          <td>0.027270</td>
          <td>0.018626</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.847606</td>
          <td>0.966765</td>
          <td>26.970794</td>
          <td>0.206894</td>
          <td>26.304894</td>
          <td>0.103229</td>
          <td>26.139360</td>
          <td>0.144590</td>
          <td>25.683521</td>
          <td>0.183151</td>
          <td>26.958963</td>
          <td>0.943929</td>
          <td>0.012691</td>
          <td>0.006402</td>
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
