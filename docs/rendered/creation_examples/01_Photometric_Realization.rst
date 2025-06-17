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

    <pzflow.flow.Flow at 0x7f337cd1ab60>



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
    0      23.994413  0.064366  0.038777  
    1      25.391064  0.015961  0.015082  
    2      24.304707  0.071496  0.066199  
    3      25.291103  0.135624  0.111757  
    4      25.096743  0.180104  0.130374  
    ...          ...       ...       ...  
    99995  24.737946  0.138279  0.099564  
    99996  24.224169  0.067645  0.034889  
    99997  25.613836  0.171345  0.119498  
    99998  25.274899  0.117946  0.111214  
    99999  25.699642  0.037842  0.032978  
    
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
          <td>28.693847</td>
          <td>1.539147</td>
          <td>26.512682</td>
          <td>0.140013</td>
          <td>26.138664</td>
          <td>0.089096</td>
          <td>25.211952</td>
          <td>0.064072</td>
          <td>24.745034</td>
          <td>0.081076</td>
          <td>24.048749</td>
          <td>0.098642</td>
          <td>0.064366</td>
          <td>0.038777</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.602874</td>
          <td>0.720267</td>
          <td>26.564562</td>
          <td>0.129241</td>
          <td>26.347435</td>
          <td>0.172513</td>
          <td>25.888631</td>
          <td>0.217304</td>
          <td>25.793113</td>
          <td>0.420412</td>
          <td>0.015961</td>
          <td>0.015082</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.576626</td>
          <td>2.264560</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.151346</td>
          <td>0.471728</td>
          <td>26.034822</td>
          <td>0.131934</td>
          <td>25.140447</td>
          <td>0.114684</td>
          <td>24.203662</td>
          <td>0.112946</td>
          <td>0.071496</td>
          <td>0.066199</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.734293</td>
          <td>0.383145</td>
          <td>27.460892</td>
          <td>0.274970</td>
          <td>26.111146</td>
          <td>0.140921</td>
          <td>25.605647</td>
          <td>0.171217</td>
          <td>25.465159</td>
          <td>0.325559</td>
          <td>0.135624</td>
          <td>0.111757</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.143098</td>
          <td>0.284075</td>
          <td>26.086428</td>
          <td>0.096660</td>
          <td>25.971755</td>
          <td>0.076904</td>
          <td>25.619933</td>
          <td>0.091867</td>
          <td>25.449451</td>
          <td>0.149824</td>
          <td>25.116736</td>
          <td>0.245525</td>
          <td>0.180104</td>
          <td>0.130374</td>
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
          <td>27.269736</td>
          <td>0.663742</td>
          <td>26.287754</td>
          <td>0.115238</td>
          <td>25.468088</td>
          <td>0.049208</td>
          <td>25.091846</td>
          <td>0.057596</td>
          <td>24.834127</td>
          <td>0.087697</td>
          <td>24.960578</td>
          <td>0.215715</td>
          <td>0.138279</td>
          <td>0.099564</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.879251</td>
          <td>0.502446</td>
          <td>26.795324</td>
          <td>0.178265</td>
          <td>26.050947</td>
          <td>0.082472</td>
          <td>25.282410</td>
          <td>0.068200</td>
          <td>24.785304</td>
          <td>0.084007</td>
          <td>24.313940</td>
          <td>0.124314</td>
          <td>0.067645</td>
          <td>0.034889</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.597931</td>
          <td>1.467277</td>
          <td>26.596212</td>
          <td>0.150433</td>
          <td>26.407385</td>
          <td>0.112743</td>
          <td>26.475644</td>
          <td>0.192289</td>
          <td>25.899645</td>
          <td>0.219308</td>
          <td>25.667533</td>
          <td>0.381675</td>
          <td>0.171345</td>
          <td>0.119498</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.009470</td>
          <td>0.552469</td>
          <td>26.231929</td>
          <td>0.109769</td>
          <td>26.276200</td>
          <td>0.100530</td>
          <td>25.957662</td>
          <td>0.123402</td>
          <td>25.742565</td>
          <td>0.192262</td>
          <td>25.083771</td>
          <td>0.238941</td>
          <td>0.117946</td>
          <td>0.111214</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.792853</td>
          <td>1.614966</td>
          <td>26.688756</td>
          <td>0.162823</td>
          <td>26.608607</td>
          <td>0.134260</td>
          <td>26.465325</td>
          <td>0.190624</td>
          <td>26.054756</td>
          <td>0.249346</td>
          <td>25.784232</td>
          <td>0.417570</td>
          <td>0.037842</td>
          <td>0.032978</td>
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
          <td>27.275447</td>
          <td>0.733578</td>
          <td>26.486898</td>
          <td>0.158797</td>
          <td>25.866632</td>
          <td>0.083247</td>
          <td>25.175962</td>
          <td>0.074295</td>
          <td>24.705358</td>
          <td>0.092931</td>
          <td>24.179839</td>
          <td>0.131753</td>
          <td>0.064366</td>
          <td>0.038777</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.417026</td>
          <td>2.242217</td>
          <td>28.164768</td>
          <td>0.595713</td>
          <td>26.377546</td>
          <td>0.128998</td>
          <td>26.073232</td>
          <td>0.160927</td>
          <td>25.726980</td>
          <td>0.221486</td>
          <td>25.613711</td>
          <td>0.424079</td>
          <td>0.015961</td>
          <td>0.015082</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.797451</td>
          <td>0.527135</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.001628</td>
          <td>0.490350</td>
          <td>26.126899</td>
          <td>0.171030</td>
          <td>25.121507</td>
          <td>0.134405</td>
          <td>24.295853</td>
          <td>0.146554</td>
          <td>0.071496</td>
          <td>0.066199</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.101498</td>
          <td>0.542712</td>
          <td>26.190191</td>
          <td>0.186664</td>
          <td>25.778911</td>
          <td>0.242148</td>
          <td>25.984779</td>
          <td>0.581739</td>
          <td>0.135624</td>
          <td>0.111757</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.113284</td>
          <td>0.325997</td>
          <td>26.249689</td>
          <td>0.137578</td>
          <td>26.023237</td>
          <td>0.102204</td>
          <td>25.654202</td>
          <td>0.121164</td>
          <td>25.354425</td>
          <td>0.174169</td>
          <td>25.548400</td>
          <td>0.431775</td>
          <td>0.180104</td>
          <td>0.130374</td>
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
          <td>27.968124</td>
          <td>1.149315</td>
          <td>26.329260</td>
          <td>0.143398</td>
          <td>25.549999</td>
          <td>0.065346</td>
          <td>25.164687</td>
          <td>0.076476</td>
          <td>25.065311</td>
          <td>0.131993</td>
          <td>25.059841</td>
          <td>0.286186</td>
          <td>0.138279</td>
          <td>0.099564</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.550050</td>
          <td>0.436862</td>
          <td>26.748618</td>
          <td>0.198259</td>
          <td>26.011826</td>
          <td>0.094607</td>
          <td>25.247020</td>
          <td>0.079125</td>
          <td>24.576883</td>
          <td>0.083020</td>
          <td>24.398740</td>
          <td>0.159076</td>
          <td>0.067645</td>
          <td>0.034889</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>37.382696</td>
          <td>10.120237</td>
          <td>27.045474</td>
          <td>0.266625</td>
          <td>26.049039</td>
          <td>0.103659</td>
          <td>26.085588</td>
          <td>0.174051</td>
          <td>25.477989</td>
          <td>0.191791</td>
          <td>25.781040</td>
          <td>0.509979</td>
          <td>0.171345</td>
          <td>0.119498</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.183469</td>
          <td>0.336381</td>
          <td>26.018399</td>
          <td>0.109157</td>
          <td>26.072955</td>
          <td>0.103163</td>
          <td>26.369835</td>
          <td>0.215547</td>
          <td>26.196533</td>
          <td>0.337208</td>
          <td>24.573913</td>
          <td>0.190742</td>
          <td>0.117946</td>
          <td>0.111214</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.376853</td>
          <td>0.143842</td>
          <td>26.644483</td>
          <td>0.162828</td>
          <td>26.382865</td>
          <td>0.209809</td>
          <td>25.340795</td>
          <td>0.160432</td>
          <td>25.155837</td>
          <td>0.297068</td>
          <td>0.037842</td>
          <td>0.032978</td>
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
          <td>26.432283</td>
          <td>0.134800</td>
          <td>26.117662</td>
          <td>0.090735</td>
          <td>25.258651</td>
          <td>0.069419</td>
          <td>24.814329</td>
          <td>0.089403</td>
          <td>24.078159</td>
          <td>0.105125</td>
          <td>0.064366</td>
          <td>0.038777</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.641326</td>
          <td>0.421184</td>
          <td>27.295040</td>
          <td>0.270847</td>
          <td>26.655277</td>
          <td>0.140227</td>
          <td>26.368210</td>
          <td>0.176167</td>
          <td>25.559827</td>
          <td>0.165186</td>
          <td>25.033101</td>
          <td>0.229864</td>
          <td>0.015961</td>
          <td>0.015082</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.332241</td>
          <td>1.173625</td>
          <td>27.713931</td>
          <td>0.355646</td>
          <td>25.946408</td>
          <td>0.130091</td>
          <td>24.959326</td>
          <td>0.103978</td>
          <td>24.588815</td>
          <td>0.167448</td>
          <td>0.071496</td>
          <td>0.066199</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.310015</td>
          <td>0.664521</td>
          <td>27.075751</td>
          <td>0.235665</td>
          <td>26.416919</td>
          <td>0.217430</td>
          <td>25.053139</td>
          <td>0.126158</td>
          <td>24.998390</td>
          <td>0.263291</td>
          <td>0.135624</td>
          <td>0.111757</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.769395</td>
          <td>0.541637</td>
          <td>26.067476</td>
          <td>0.118810</td>
          <td>26.025134</td>
          <td>0.103588</td>
          <td>25.865303</td>
          <td>0.147141</td>
          <td>25.363457</td>
          <td>0.177524</td>
          <td>24.613623</td>
          <td>0.206305</td>
          <td>0.180104</td>
          <td>0.130374</td>
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
          <td>27.213239</td>
          <td>0.700442</td>
          <td>26.370133</td>
          <td>0.142703</td>
          <td>25.448218</td>
          <td>0.057036</td>
          <td>25.131476</td>
          <td>0.070838</td>
          <td>24.885355</td>
          <td>0.107943</td>
          <td>25.030081</td>
          <td>0.267599</td>
          <td>0.138279</td>
          <td>0.099564</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.278924</td>
          <td>0.681767</td>
          <td>26.652628</td>
          <td>0.162963</td>
          <td>26.036936</td>
          <td>0.084575</td>
          <td>25.216975</td>
          <td>0.066955</td>
          <td>24.827419</td>
          <td>0.090505</td>
          <td>24.225933</td>
          <td>0.119669</td>
          <td>0.067645</td>
          <td>0.034889</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.104927</td>
          <td>0.675911</td>
          <td>26.291179</td>
          <td>0.141085</td>
          <td>26.627820</td>
          <td>0.170270</td>
          <td>26.399814</td>
          <td>0.225804</td>
          <td>25.639142</td>
          <td>0.218762</td>
          <td>26.559033</td>
          <td>0.866999</td>
          <td>0.171345</td>
          <td>0.119498</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.643753</td>
          <td>0.462874</td>
          <td>26.172416</td>
          <td>0.119053</td>
          <td>26.218124</td>
          <td>0.111133</td>
          <td>25.898584</td>
          <td>0.137032</td>
          <td>25.790981</td>
          <td>0.231238</td>
          <td>25.407274</td>
          <td>0.358053</td>
          <td>0.117946</td>
          <td>0.111214</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.588752</td>
          <td>1.470334</td>
          <td>26.853912</td>
          <td>0.189968</td>
          <td>26.760826</td>
          <td>0.155586</td>
          <td>26.313020</td>
          <td>0.170424</td>
          <td>26.185711</td>
          <td>0.281842</td>
          <td>26.874300</td>
          <td>0.905858</td>
          <td>0.037842</td>
          <td>0.032978</td>
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
