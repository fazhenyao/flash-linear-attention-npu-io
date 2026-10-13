/* Browser-only XLSX export. ZIP entries use STORE so Pages needs no CDN or server. */
(function (root) {
  "use strict";
  const MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  const NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
  const finite = (v) => v !== null && v !== undefined && typeof v !== "boolean" && String(v).trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null;
  const column = (n) => { let s = ""; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
  const xml = (v) => String(v ?? "").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
  const caseName = (name) => /^case\d{1,6}$/i.test(name || "") ? name.toLowerCase() : "";
  const textHeight = (text, width = 95) => Math.min(409, String(text).split("\n").reduce((n, line) => n + Math.max(1, Math.ceil(line.length / width)), 0) * 13 + 4);
  const status = (v) => ({ done: "", succeeded: "", passed: "", reconstructed: "", failed: "运行失败", unsupported: "不支持", missing: "缺少结果", canceled: "已取消", orphaned: "结果无法确认" })[v] ?? (v ? "尚未完成" : "");

  function durationField(fields) {
    for (const unit of ["us", "µs", "μs", "ns", "ms", "s"]) {
      const name = fields.find((f) => f.replace(/\s/g, "").toLowerCase() === `taskduration(${unit})`);
      if (name) return { name, multiplier: ({ ns: .001, us: 1, "µs": 1, "μs": 1, ms: 1000, s: 1000000 })[unit] };
    }
    return { name: "Task Duration(us)", multiplier: 1 };
  }

  function executionTasks(record) {
    const snap = record.snapshot || {};
    const details = snap.op_summary_details;
    if (details && Array.isArray(details.rows) && Array.isArray(details.names) && Array.isArray(details.fields)) {
      const field = durationField(details.fields), index = details.fields.indexOf(field.name);
      return { kind: "op_summary 逐条设备任务耗时（非重复采样均值）", tasks: details.rows.map((row) => {
        const name = details.names[row[0]];
        if (!name || !name.op_name) throw new Error("采集记录包含无效的小算子名称索引，请重新导入 Prof。");
        const value = index >= 0 ? finite(row[index + 1]) : null;
        return { ...name, time: value !== null && value >= 0 ? value * field.multiplier : null };
      }) };
    }
    if (Array.isArray(snap.op_summary)) return {
      kind: "同名汇总，缺少逐条明细；耗时为累计值",
      tasks: snap.op_summary.map((op) => {
        const field = durationField(Object.keys(op.time_stats || {})), value = finite(op.time_stats?.[field.name]?.total);
        return { ...op, op_name: `${op.op_name}【同名汇总】`, time: value !== null && value >= 0 ? value * field.multiplier : null };
      }),
    };
    return { kind: "历史逻辑汇总，缺少原始小算子名称、执行顺序和输入 shape", tasks: (snap.operators || []).map((op) => {
      const value = finite(op.time_ms);
      return { op_name: `历史逻辑汇总：${op.operator_id || "未命名"}`, time: value !== null && value >= 0 ? value * 1000 : null };
    }) };
  }

  function taskGroup(task, record) {
    if (task.block || task.sub_block) return [task.block || "未记录 Block", task.sub_block || "执行明细"];
    const name = task.op_name || "";
    const known = [
      [/ChunkGatedDeltaRule(?:Fwd|Bwd)/i, "chunk_gdn"], [/ChunkKda(?:Fwd|Bwd)/i, "chunk_kda"],
      [/RecurrentGatedDeltaRule/i, "recurrent_gdn"], [/RecurrentKda/i, "recurrent_kda"], [/CausalConv1d/i, "causal_conv1d"],
    ].find(([pattern]) => pattern.test(name));
    const block = known?.[1] || record.example || record.caseLabel || "设备任务";
    const direction = record.direction || (/Bwd|Backward/i.test(name) ? "backward" : /Fwd|Forward|Recurrent|^CausalConv1d$/i.test(name) ? "forward" : "");
    return [block, direction === "backward" ? "反向" : direction === "forward" ? "正向" : "执行明细"];
  }

  function buildWorkbook(records) {
    if (!records.length) throw new Error("请至少选择一条历史测试记录。");
    const cases = new Map();
    for (const record of records) {
      const key = record.sheetGroup ? `sheet:${caseName(record.sheetGroup) || record.sheetGroup}` : record.caseKey || record.caseId || record.id;
      if (!cases.has(key)) cases.set(key, []);
      cases.get(key).push(record);
    }
    const reserved = new Set(records.map((r) => caseName(r.sheetGroup)).filter(Boolean));
    let nextCase = 0;
    const sheets = [], provenance = [["sheet", "采集列组", "历史记录 ID", "采集时间", "case ID", "测试示例", "平台 / 芯片", "版本 / 分支", "输入参数（记录值）", "时间数据口径", "采集来源", "结果状态", "case 分组（选择值）"]];
    for (const captures of cases.values()) {
      // Disjoint operator sections from the same platform/version share a pair
      // of columns. Repeated collections of the same section stay independent.
      const collections = [];
      for (const record of captures) {
        const collection = executionTasks(record), groups = new Map();
        for (const task of collection.tasks) {
          const key = JSON.stringify(taskGroup(task, record));
          if (!groups.has(key)) groups.set(key, { record, kind: collection.kind, tasks: [] });
          groups.get(key).tasks.push(task);
        }
        if (!groups.size) groups.set(JSON.stringify([record.example || "测试记录", record.direction === "backward" ? "反向" : record.direction === "forward" ? "正向" : "执行明细"]), { record, kind: collection.kind, tasks: [] });
        let target = collections.find((c) => c.record.platform === record.platform && c.record.version === record.version
          && [...groups.keys()].every((key) => !c.groups.has(key)));
        if (!target) { target = { record, groups: new Map(), sources: [] }; collections.push(target); }
        for (const [key, value] of groups) target.groups.set(key, value);
        target.sources.push({ record, kind: collection.kind, count: collection.tasks.length });
      }
      let name = caseName(captures[0].sheetGroup);
      if (!name) {
        while (reserved.has(`case${nextCase}`)) nextCase++;
        name = `case${nextCase++}`; reserved.add(name);
      }
      const sheet = { name, rows: [], merges: [], heights: {}, columns: [20, 13], frozenRows: 2, frozenColumns: 2 };
      const width = 2 + collections.length * 2;
      if (width > 16384) throw new Error("选择的采集列数超过 Excel 上限，请减少历史记录数量。");
      const cell = (r, c, v, style = typeof v === "number" ? 4 : 0, f = null) => {
        while (sheet.rows.length <= r) sheet.rows.push([]);
        sheet.rows[r][c] = { v, style, f };
      };
      const merge = (r1, c1, r2, c2) => { if (r1 !== r2 || c1 !== c2) sheet.merges.push(`${column(c1)}${r1 + 1}:${column(c2)}${r2 + 1}`); };
      cell(0, 0, "Block", 1); cell(0, 1, "SubBlock", 1); sheet.heights[0] = 30;
      cell(1, 0, `${sheet.name}。按记录中的 Task Duration 导出（us），不自动去预热或求采样均值。分组内保留重复任务及源顺序；不同采集同一行不代表同一任务。`, 11);
      merge(1, 0, 1, width - 1); sheet.heights[1] = 24;
      collections.forEach((c, i) => {
        const r = c.record;
        const caption = `${r.platform || "平台未记录"} ${r.version || "版本未记录"} ${c.sources.length === 1 ? r.time || "时间未记录" : `${c.sources.length} 条记录`} [${i + 1}]`;
        cell(0, 2 + i * 2, `${caption}\n算子名称`, 1); cell(0, 3 + i * 2, "耗时（us）", 1);
        sheet.columns.push(83, 17);
        c.sources.forEach(({ record: r, kind, count }) => provenance.push([sheet.name, caption, r.id, r.time, r.caseId, r.example, r.platform, r.version,
          JSON.stringify(r.attributes || {}), kind, r.source, status(r.status) || (count ? "有性能数据" : "缺少结果"), r.sheetGroup || "自动"]));
      });
      const keys = [...new Set(collections.flatMap((c) => [...c.groups.keys()]))];
      if (!keys.length) keys.push(JSON.stringify([captures[0].example || "测试记录", captures[0].direction === "backward" ? "反向" : "执行明细"]));
      let row = 2;
      const blockRanges = [];
      for (const key of keys) {
        const [block, sub] = JSON.parse(key), start = row;
        const sections = collections.map((c) => c.groups.get(key) || { record: c.record, tasks: [], absent: true });
        const lists = sections.map((c) => c.tasks), count = Math.max(1, ...lists.map((list) => list.length));
        cell(row, 1, "输入 shape", 2); sheet.heights[row] = 45;
        sections.forEach((c, i) => {
          const task = lists[i].find((t) => t.input_shapes && /Chunk|Recurrent|CausalConv/i.test(t.op_name) && !/Cast|Transpose/.test(t.op_name)) || lists[i].find((t) => t.input_shapes);
          const actual = task?.input_shapes || c.record.actualShape;
          const layout = task?.input_layout || c.record.actualLayout || "轴 layout 未记录";
          const format = task?.input_formats ? `；存储格式：${task.input_formats}` : "";
          const types = task?.input_dtypes ? `\n数据类型：${task.input_dtypes}` : "";
          const description = c.absent ? "" : actual ? `输入 shape（采集记录）：${actual}\n${layout}${format}${types}` : "实际输入 shape / layout 未记录";
          cell(row, 2 + i * 2, description, 2);
          sheet.heights[row] = Math.max(sheet.heights[row], textHeight(description));
          merge(row, 2 + i * 2, row, 3 + i * 2);
        });
        row++;
        cell(row, 1, "输入参数", 2); sheet.heights[row] = 44;
        sections.forEach((c, i) => {
          const description = c.absent ? "" : JSON.stringify(c.record.attributes || {});
          cell(row, 2 + i * 2, description, 2);
          sheet.heights[row] = Math.max(sheet.heights[row], textHeight(description));
          merge(row, 2 + i * 2, row, 3 + i * 2);
        });
        row++;
        const statusRow = row++, firstTask = row, lastTask = row + count - 1, total = lastTask + 1;
        cell(statusRow, 1, sub, 3); merge(statusRow, 1, total, 1);
        sections.forEach((c, i) => {
          const nc = 2 + i * 2, tc = nc + 1, tasks = lists[i];
          const label = c.absent ? "未选择该项" : status(c.record.status) || (!tasks.length ? "缺少结果" : "");
          if (label) cell(statusRow, nc, label, label === "运行失败" ? 10 : 9);
          merge(statusRow, nc, statusRow, tc);
          tasks.forEach((t, j) => {
            const kernel = /Chunk|Recurrent|CausalConv/i.test(t.op_name) && !/Cast|Transpose/.test(t.op_name);
            cell(firstTask + j, nc, t.op_name, kernel ? 5 : 0); cell(firstTask + j, tc, t.time, kernel ? 6 : 4);
          });
          cell(total, nc, `${sub === "执行明细" ? "记录" : sub}部分耗时`, 7);
          // Never turn missing samples or an incomplete execution into a zero/partial total.
          if (!status(c.record.status) && tasks.length && tasks.every((t) => t.time !== null)) {
            const value = Math.round(tasks.reduce((s, t) => s + t.time, 0) * 1e6) / 1e6;
            cell(total, tc, value, 8, `ROUND(SUM(${column(tc)}${firstTask + 1}:${column(tc)}${firstTask + tasks.length}),6)`);
          } else cell(total, tc, label || "缺少耗时", 7);
        });
        for (let j = 0; j < count; j++) sheet.heights[firstTask + j] = Math.max(18, Math.ceil(Math.max(...lists.map((list) => list[j]?.op_name.length || 0)) / 82) * 13 + 4);
        sheet.heights[total] = 21;
        const previous = blockRanges[blockRanges.length - 1];
        if (previous?.block === block) previous.end = total;
        else { cell(start, 0, block, 3); blockRanges.push({ block, start, end: total }); }
        row = total + 2; sheet.heights[total + 1] = 8;
      }
      for (const range of blockRanges) merge(range.start, 0, range.end, 0);
      sheets.push(sheet);
    }
    sheets.push({ name: "来源明细", rows: provenance.map((r, i) => r.map((v) => ({ v: v ?? "", style: i ? 0 : 1 }))),
      merges: [], heights: Object.fromEntries(provenance.map((_, i) => [i, i ? 45 : 30])),
      columns: [10, 65, 38, 24, 45, 28, 25, 28, 90, 70, 65, 18, 20], frozenRows: 1, frozenColumns: 0 });
    return { sheets };
  }

  function sheetXml(sheet) {
    const width = sheet.columns.length, height = sheet.rows.length;
    if (height > 1048576) throw new Error("设备任务行数超过 Excel 上限，请减少历史记录数量。");
    const rows = sheet.rows.map((row, r) => `<row r="${r + 1}" ht="${sheet.heights[r] || 18}" customHeight="1">${row.map((c, i) => {
      if (!c || c.v === null || c.v === undefined) return "";
      const a = `${column(i)}${r + 1}`, style = ` s="${c.style || 0}"`;
      if (typeof c.v === "number") {
        if (!Number.isFinite(c.v)) throw new Error("导出包含无效数值。");
        return `<c r="${a}"${style}>${c.f ? `<f>${xml(c.f)}</f>` : ""}<v>${c.v}</v></c>`;
      }
      if (String(c.v).length > 32767) throw new Error("单元格文本超过 Excel 上限，请减少输入参数内容。");
      return `<c r="${a}"${style} t="inlineStr"><is><t xml:space="preserve">${xml(c.v)}</t></is></c>`;
    }).join("")}</row>`).join("");
    const freeze = sheet.frozenRows || sheet.frozenColumns ? `<pane xSplit="${sheet.frozenColumns}" ySplit="${sheet.frozenRows}" topLeftCell="${column(sheet.frozenColumns)}${sheet.frozenRows + 1}" activePane="${sheet.frozenColumns ? "bottomRight" : "bottomLeft"}" state="frozen"/>` : "";
    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="${NS}"><dimension ref="A1:${column(width - 1)}${height}"/><sheetViews><sheetView workbookViewId="0" showGridLines="1">${freeze}</sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/><cols>${sheet.columns.map((w, i) => `<col min="${i + 1}" max="${i + 1}" width="${w}" customWidth="1"/>`).join("")}</cols><sheetData>${rows}</sheetData>${sheet.merges.length ? `<mergeCells count="${sheet.merges.length}">${sheet.merges.map((m) => `<mergeCell ref="${xml(m)}"/>`).join("")}</mergeCells>` : ""}</worksheet>`;
  }

  function stylesXml() {
    const font = (color, bold = false) => `<font><sz val="9"/><color rgb="FF${color}"/><name val="Microsoft YaHei"/>${bold ? "<b/>" : ""}</font>`;
    const xf = (fontId = 0, fillId = 0, borderId = 0, numeric = false, align = "left", wrap = true) => `<xf numFmtId="${numeric ? 164 : 0}" fontId="${fontId}" fillId="${fillId}" borderId="${borderId}" xfId="0" applyAlignment="1" applyNumberFormat="1"><alignment horizontal="${align}" vertical="center" wrapText="${wrap ? 1 : 0}"/></xf>`;
    const styles = [xf(), xf(1, 2), xf(0, 3), xf(0, 0, 0, false, "center"), xf(0, 0, 0, true, "right", false),
      xf(2), xf(2, 0, 0, true, "right", false), xf(2, 0, 1, false, "center"), xf(2, 0, 1, true, "right", false),
      xf(3, 4), xf(4, 5), xf(5)];
    return `<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="${NS}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts><fonts count="6">${font("000000")}${font("000000", true)}${font("FF0000")}${font("9C6500")}${font("9C0006", true)}${font("595959")}</fonts><fills count="6"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>${["D0CECE", "F2F2F2", "FFF2CC", "FCE4D6"].map((c) => `<fill><patternFill patternType="solid"><fgColor rgb="FF${c}"/><bgColor indexed="64"/></patternFill></fill>`).join("")}</fills><borders count="2"><border/><border><top style="thin"><color rgb="FFA6A6A6"/></top><bottom style="thin"><color rgb="FFA6A6A6"/></bottom></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="${styles.length}">${styles.join("")}</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>`;
  }

  // Standard ZIP32 container with UTF-8 names, CRC32 and an uncompressed payload.
  function zip(files) {
    const encoder = new TextEncoder(), local = [], central = [];
    const table = Uint32Array.from({ length: 256 }, (_, n) => {
      for (let i = 0; i < 8; i++) n = n & 1 ? 0xEDB88320 ^ (n >>> 1) : n >>> 1;
      return n >>> 0;
    });
    const header = (length, fields) => {
      const bytes = new Uint8Array(length), view = new DataView(bytes.buffer);
      for (const [offset, value, size] of fields) size === 2 ? view.setUint16(offset, value, true) : view.setUint32(offset, value, true);
      return bytes;
    };
    let offset = 0;
    for (const [path, text] of files) {
      const name = encoder.encode(path), data = encoder.encode(text);
      let crc = 0xFFFFFFFF; for (const byte of data) crc = table[(crc ^ byte) & 255] ^ (crc >>> 8); crc = (crc ^ 0xFFFFFFFF) >>> 0;
      local.push(header(30, [[0, 0x04034B50, 4], [4, 20, 2], [6, 0x800, 2], [12, 33, 2], [14, crc, 4], [18, data.length, 4], [22, data.length, 4], [26, name.length, 2]]), name, data);
      central.push(header(46, [[0, 0x02014B50, 4], [4, 20, 2], [6, 20, 2], [8, 0x800, 2], [14, 33, 2], [16, crc, 4], [20, data.length, 4], [24, data.length, 4], [28, name.length, 2], [42, offset, 4]]), name);
      offset += 30 + name.length + data.length;
    }
    const size = central.reduce((sum, b) => sum + b.length, 0);
    if (offset + size > 0xFFFFFFFF || files.length > 65535) throw new Error("导出超过 ZIP32 上限，请减少历史记录数量。");
    const parts = [...local, ...central, header(22, [[0, 0x06054B50, 4], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, offset, 4]])];
    const output = new Uint8Array(parts.reduce((sum, b) => sum + b.length, 0));
    let position = 0; for (const part of parts) { output.set(part, position); position += part.length; }
    return output;
  }

  function writeXlsx(workbook) {
    const sheets = workbook.sheets;
    const rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    const files = [
      ["[Content_Types].xml", `<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>${sheets.map((_, i) => `<Override PartName="/xl/worksheets/sheet${i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`).join("")}</Types>`],
      ["_rels/.rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>`],
      ["xl/workbook.xml", `<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="${NS}" xmlns:r="${rel}"><bookViews><workbookView/></bookViews><sheets>${sheets.map((s, i) => `<sheet name="${xml(s.name)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`).join("")}</sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>`],
      ["xl/_rels/workbook.xml.rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${sheets.map((_, i) => `<Relationship Id="rId${i + 1}" Type="${rel}/worksheet" Target="worksheets/sheet${i + 1}.xml"/>`).join("")}<Relationship Id="rId${sheets.length + 1}" Type="${rel}/styles" Target="styles.xml"/></Relationships>`],
      ["xl/styles.xml", stylesXml()],
      ...sheets.map((s, i) => [`xl/worksheets/sheet${i + 1}.xml`, sheetXml(s)]),
    ];
    return zip(files);
  }
  root.PerfXlsxExport = { MIME, executionTasks, buildWorkbook, writeXlsx };
})(globalThis);
