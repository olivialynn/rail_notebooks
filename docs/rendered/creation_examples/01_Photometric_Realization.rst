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

    <pzflow.flow.Flow at 0x7fa34dfeba00>



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
          <td>28.196792</td>
          <td>1.185085</td>
          <td>26.797139</td>
          <td>0.178540</td>
          <td>26.008182</td>
          <td>0.079418</td>
          <td>25.223206</td>
          <td>0.064714</td>
          <td>24.806382</td>
          <td>0.085581</td>
          <td>23.733025</td>
          <td>0.074701</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.250234</td>
          <td>0.260438</td>
          <td>26.699904</td>
          <td>0.145254</td>
          <td>26.283620</td>
          <td>0.163386</td>
          <td>26.016688</td>
          <td>0.241651</td>
          <td>25.512671</td>
          <td>0.338060</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.890753</td>
          <td>0.386890</td>
          <td>26.081393</td>
          <td>0.137351</td>
          <td>25.292938</td>
          <td>0.130918</td>
          <td>24.213233</td>
          <td>0.113892</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.277435</td>
          <td>2.007364</td>
          <td>30.694385</td>
          <td>2.181853</td>
          <td>27.554025</td>
          <td>0.296492</td>
          <td>26.284604</td>
          <td>0.163524</td>
          <td>25.515350</td>
          <td>0.158527</td>
          <td>25.518327</td>
          <td>0.339575</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.085402</td>
          <td>0.271093</td>
          <td>26.075777</td>
          <td>0.095762</td>
          <td>25.905540</td>
          <td>0.072532</td>
          <td>25.744652</td>
          <td>0.102487</td>
          <td>25.404148</td>
          <td>0.144103</td>
          <td>25.073527</td>
          <td>0.236927</td>
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
          <td>26.465708</td>
          <td>0.367096</td>
          <td>26.627890</td>
          <td>0.154572</td>
          <td>25.372474</td>
          <td>0.045203</td>
          <td>25.048668</td>
          <td>0.055431</td>
          <td>24.790283</td>
          <td>0.084376</td>
          <td>24.712512</td>
          <td>0.175059</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.360051</td>
          <td>0.337873</td>
          <td>26.946433</td>
          <td>0.202485</td>
          <td>26.215748</td>
          <td>0.095340</td>
          <td>25.069664</td>
          <td>0.056473</td>
          <td>24.820857</td>
          <td>0.086679</td>
          <td>24.314511</td>
          <td>0.124376</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.401359</td>
          <td>0.349054</td>
          <td>26.528313</td>
          <td>0.141910</td>
          <td>26.460930</td>
          <td>0.118124</td>
          <td>26.065566</td>
          <td>0.135487</td>
          <td>25.959200</td>
          <td>0.230433</td>
          <td>26.135550</td>
          <td>0.542475</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.336285</td>
          <td>0.331581</td>
          <td>26.109750</td>
          <td>0.098655</td>
          <td>26.117046</td>
          <td>0.087417</td>
          <td>26.050071</td>
          <td>0.133686</td>
          <td>25.988072</td>
          <td>0.236007</td>
          <td>25.147165</td>
          <td>0.251746</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.950648</td>
          <td>0.529415</td>
          <td>26.620245</td>
          <td>0.153563</td>
          <td>26.703379</td>
          <td>0.145689</td>
          <td>26.347952</td>
          <td>0.172589</td>
          <td>26.046511</td>
          <td>0.247661</td>
          <td>26.253128</td>
          <td>0.590227</td>
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
          <td>26.764960</td>
          <td>0.199322</td>
          <td>25.927410</td>
          <td>0.086992</td>
          <td>25.320600</td>
          <td>0.083580</td>
          <td>24.761436</td>
          <td>0.096698</td>
          <td>23.884525</td>
          <td>0.100913</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.132497</td>
          <td>2.888606</td>
          <td>27.692548</td>
          <td>0.420530</td>
          <td>26.611166</td>
          <td>0.157638</td>
          <td>26.238245</td>
          <td>0.185044</td>
          <td>25.713729</td>
          <td>0.218935</td>
          <td>25.114952</td>
          <td>0.286335</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.223905</td>
          <td>0.713962</td>
          <td>29.672126</td>
          <td>1.502126</td>
          <td>28.126618</td>
          <td>0.540287</td>
          <td>26.611668</td>
          <td>0.258058</td>
          <td>24.985806</td>
          <td>0.120257</td>
          <td>24.113149</td>
          <td>0.125987</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.048855</td>
          <td>1.214745</td>
          <td>30.345809</td>
          <td>2.081393</td>
          <td>27.138932</td>
          <td>0.261461</td>
          <td>26.588192</td>
          <td>0.264292</td>
          <td>25.204438</td>
          <td>0.151778</td>
          <td>25.207872</td>
          <td>0.328410</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.412935</td>
          <td>0.390938</td>
          <td>26.286601</td>
          <td>0.132620</td>
          <td>25.890337</td>
          <td>0.084226</td>
          <td>25.589528</td>
          <td>0.105867</td>
          <td>25.231682</td>
          <td>0.145553</td>
          <td>25.620644</td>
          <td>0.426148</td>
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
          <td>28.374760</td>
          <td>1.414706</td>
          <td>26.852270</td>
          <td>0.218323</td>
          <td>25.534814</td>
          <td>0.062810</td>
          <td>24.993999</td>
          <td>0.064011</td>
          <td>24.872826</td>
          <td>0.108849</td>
          <td>24.780244</td>
          <td>0.221945</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.804919</td>
          <td>0.526100</td>
          <td>26.481830</td>
          <td>0.157376</td>
          <td>26.015635</td>
          <td>0.094399</td>
          <td>25.346021</td>
          <td>0.085844</td>
          <td>24.897326</td>
          <td>0.109358</td>
          <td>24.510044</td>
          <td>0.173938</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.588202</td>
          <td>0.450535</td>
          <td>26.793564</td>
          <td>0.206420</td>
          <td>26.556060</td>
          <td>0.152228</td>
          <td>26.094949</td>
          <td>0.165925</td>
          <td>25.522582</td>
          <td>0.188764</td>
          <td>25.258121</td>
          <td>0.324991</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.984165</td>
          <td>0.284559</td>
          <td>26.356628</td>
          <td>0.144807</td>
          <td>26.021365</td>
          <td>0.097494</td>
          <td>25.623348</td>
          <td>0.112580</td>
          <td>25.507231</td>
          <td>0.189695</td>
          <td>25.600242</td>
          <td>0.431336</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.079698</td>
          <td>0.302719</td>
          <td>26.469996</td>
          <td>0.156597</td>
          <td>26.466570</td>
          <td>0.140580</td>
          <td>25.966307</td>
          <td>0.148221</td>
          <td>26.214777</td>
          <td>0.332179</td>
          <td>25.923878</td>
          <td>0.538517</td>
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
          <td>26.945560</td>
          <td>0.527496</td>
          <td>26.588102</td>
          <td>0.149407</td>
          <td>25.913174</td>
          <td>0.073033</td>
          <td>25.183219</td>
          <td>0.062469</td>
          <td>24.708483</td>
          <td>0.078513</td>
          <td>23.961463</td>
          <td>0.091379</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.692318</td>
          <td>1.538554</td>
          <td>27.048930</td>
          <td>0.220757</td>
          <td>26.813944</td>
          <td>0.160321</td>
          <td>26.050674</td>
          <td>0.133886</td>
          <td>25.803481</td>
          <td>0.202550</td>
          <td>25.035013</td>
          <td>0.229705</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.437667</td>
          <td>0.378288</td>
          <td>32.246742</td>
          <td>3.696357</td>
          <td>27.817989</td>
          <td>0.393017</td>
          <td>26.020559</td>
          <td>0.141745</td>
          <td>25.028070</td>
          <td>0.112775</td>
          <td>24.263307</td>
          <td>0.129354</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.832768</td>
          <td>0.557488</td>
          <td>29.315513</td>
          <td>1.275814</td>
          <td>27.074739</td>
          <td>0.247227</td>
          <td>26.127666</td>
          <td>0.179481</td>
          <td>25.222680</td>
          <td>0.153633</td>
          <td>25.214743</td>
          <td>0.329117</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.817783</td>
          <td>0.480503</td>
          <td>26.261816</td>
          <td>0.112803</td>
          <td>25.964433</td>
          <td>0.076518</td>
          <td>25.493110</td>
          <td>0.082284</td>
          <td>25.070974</td>
          <td>0.108093</td>
          <td>24.993011</td>
          <td>0.221934</td>
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
          <td>26.444825</td>
          <td>0.379496</td>
          <td>26.474690</td>
          <td>0.144986</td>
          <td>25.445611</td>
          <td>0.052241</td>
          <td>25.114778</td>
          <td>0.063887</td>
          <td>24.925559</td>
          <td>0.102797</td>
          <td>25.243156</td>
          <td>0.293556</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.163377</td>
          <td>0.622095</td>
          <td>26.631584</td>
          <td>0.157240</td>
          <td>26.101347</td>
          <td>0.087654</td>
          <td>25.224093</td>
          <td>0.065906</td>
          <td>24.792145</td>
          <td>0.085922</td>
          <td>24.325406</td>
          <td>0.127695</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.807953</td>
          <td>0.965842</td>
          <td>26.610058</td>
          <td>0.158700</td>
          <td>26.575886</td>
          <td>0.136949</td>
          <td>26.261684</td>
          <td>0.168532</td>
          <td>26.085329</td>
          <td>0.267572</td>
          <td>25.282970</td>
          <td>0.294680</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.252186</td>
          <td>0.334077</td>
          <td>26.149217</td>
          <td>0.112901</td>
          <td>25.935852</td>
          <td>0.083612</td>
          <td>25.718323</td>
          <td>0.112832</td>
          <td>25.740447</td>
          <td>0.214092</td>
          <td>25.154890</td>
          <td>0.282830</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.621499</td>
          <td>0.423963</td>
          <td>26.795976</td>
          <td>0.184287</td>
          <td>26.527434</td>
          <td>0.130046</td>
          <td>26.191602</td>
          <td>0.157143</td>
          <td>25.853468</td>
          <td>0.218951</td>
          <td>26.249510</td>
          <td>0.608163</td>
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
